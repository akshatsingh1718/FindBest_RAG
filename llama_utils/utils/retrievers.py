from llama_index import (
    VectorStoreIndex,
    ServiceContext,
    load_index_from_storage,
    StorageContext,
    Document,
    SimpleDirectoryReader,
)
from llama_index.llms import OpenAI
from llama_index.indices.postprocessor import (
    MetadataReplacementPostProcessor,
    SentenceTransformerRerank,
)
from llama_index.node_parser import (
    HierarchicalNodeParser,
    get_leaf_nodes,
    SentenceWindowNodeParser,
)
from llama_index.retrievers import AutoMergingRetriever
from llama_index.query_engine import RetrieverQueryEngine
import os
from typing import List, Optional, Union
from llama_utils.utils.common import (
    SENTENCE_WINDOW_INDEX_DEFAULT_DIR,
    AUTO_MERGING_INDEX_DEFAULT_DIR,
)
from pydantic import BaseModel
from abc import ABC, abstractmethod


class BaseRetrievalPipeline(ABC, BaseModel):
    @abstractmethod
    def from_default_query_engine(*args, **kwargs):
        pass

    @abstractmethod
    def from_default_index(*args, **kwargs):
        pass

    @abstractmethod
    def from_query_engine(*args, **kwargs):
        pass

    def resolve_documents(documents: Union[str, List[str], Document, List[Document]]):
        # if documents is a single path
        if isinstance(documents, str):
            documents = [documents]

            documents = SimpleDirectoryReader(input_files=documents).load_data()
            documents = Document(text="\n\n".join([doc.text for doc in documents]))

        # if docuemnts is a list of path strings
        if all(isinstance(doc, str) for doc in documents):
            documents = SimpleDirectoryReader(input_files=documents).load_data()
            documents = Document(text="\n\n".join([doc.text for doc in documents]))

        # if single document
        if isinstance(documents, Document):
            documents = [documents]

        # if documents is a list of documents
        if all(isinstance(doc, Document) for doc in documents):
            documents = Document(text="\n\n".join([doc.text for doc in documents]))
            return [documents]
        else:
            raise TypeError(f"Given documents type are not supported.")


class AutoMergingRetrievalPipeline(BaseRetrievalPipeline):
    def __init__(
        cls,
        llm,
        embed_model="local:BAAI/bge-small-en-v1.5",
        persist_dir: str = "merging_index",
        chunk_sizes=None,
        similarity_top_k=12,
        rerank_top_n=6,
        rerank_model="BAAI/bge-reranker-base",
        verbose=False,
    ):
        cls.llm = llm
        cls.embed_model = embed_model
        cls.persist_dir = persist_dir
        cls.chunk_sizes = chunk_sizes
        cls.similarity_top_k = similarity_top_k
        cls.rerank_top_n = rerank_top_n
        cls.rerank_model = rerank_model
        cls.verbose = verbose

    @staticmethod
    def from_default_query_engine(
        llm=OpenAI(model="gpt-3.5-turbo"),
        embed_model="local:BAAI/bge-small-en-v1.5",
        chunk_sizes: List[int] = [2048, 512, 128],
        persist_dir: str = "",
        similarity_top_k: int = 6,
        rerank_top_n: int = 2,
        rerank_model="BAAI/bge-reranker-base",
        documents: List[Document] = None,
        verbose=False,
    ):
        automerging_index = AutoMergingRetrievalPipeline.from_default_index(
            llm=llm,
            persist_dir=persist_dir,
            chunk_sizes=chunk_sizes,
            verbose=verbose,
            documents=documents,
        )

        query_engine = AutoMergingRetrievalPipeline.from_query_engine(
            index=automerging_index,
            similarity_top_k=similarity_top_k,
            rerank_top_n=rerank_top_n,
            rerank_model=rerank_model,
            verbose=verbose,
        )

        return query_engine

    @staticmethod
    def from_default_index(
        llm=None,
        persist_dir: str = "",
        documents: List[Document] = None,
        chunk_sizes: List[int] = [2048, 512, 128],
        verbose: bool = False,
        *args,
        **kwargs,
    ):
        embed_model = kwargs.pop("embed_model", "local:BAAI/bge-small-en-v1.5")

        llm = llm or OpenAI(model="gpt-3.5-turbo")

        merging_context = ServiceContext.from_defaults(
            llm=llm,
            embed_model=embed_model,
        )

        persist_dir = persist_dir or AUTO_MERGING_INDEX_DEFAULT_DIR

        if not os.path.exists(persist_dir):
            documents = AutoMergingRetrievalPipeline.resolve_documents(documents)
            node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=chunk_sizes)

            nodes = node_parser.get_nodes_from_documents(documents)
            leaf_nodes = get_leaf_nodes(nodes)

            storage_context = StorageContext.from_defaults()
            storage_context.docstore.add_documents(nodes)

            automerging_index = VectorStoreIndex(
                leaf_nodes,
                storage_context=storage_context,
                service_context=merging_context,
            )
            automerging_index.storage_context.persist(persist_dir=persist_dir)
        else:
            if verbose:
                print(f"Loading sentence index from {persist_dir}")

            automerging_index = load_index_from_storage(
                StorageContext.from_defaults(persist_dir=persist_dir),
                service_context=merging_context,
            )

        return automerging_index

    @staticmethod
    def from_query_engine(
        index: VectorStoreIndex,
        similarity_top_k: int = 12,
        rerank_top_n: int = 6,
        rerank_model="BAAI/bge-reranker-base",
        verbose=False,
        *args,
        **kwargs,
    ) -> RetrieverQueryEngine:
        base_retriever = index.as_retriever(similarity_top_k=similarity_top_k)
        retriever = AutoMergingRetriever(
            base_retriever, index.storage_context, verbose=True
        )
        rerank = SentenceTransformerRerank(top_n=rerank_top_n, model=rerank_model)
        auto_merging_engine = RetrieverQueryEngine.from_args(
            retriever, node_postprocessors=[rerank]
        )
        return auto_merging_engine


class SentenceWindowRetrievalPipeline(BaseRetrievalPipeline):
    def __init__(
        cls,
        llm,
        embed_model="local:BAAI/bge-small-en-v1.5",
        sentence_window_size=3,
        persist_dir="sentence_index",
        similarity_top_k=6,
        rerank_top_n=2,
        rerank_model="BAAI/bge-reranker-base",
        verbose=False,
    ):
        cls.llm = llm
        cls.embed_model = embed_model
        cls.sentence_window_size = sentence_window_size
        cls.persist_dir = persist_dir
        cls.similarity_top_k = similarity_top_k
        cls.rerank_top_n = rerank_top_n
        cls.verbose = verbose
        cls.rerank_model = rerank_model

    @staticmethod
    def from_default_query_engine(
        llm=OpenAI(model="gpt-3.5-turbo"),
        embed_model="local:BAAI/bge-small-en-v1.5",
        sentence_window_size=3,
        persist_dir="",
        similarity_top_k=6,
        rerank_top_n=2,
        rerank_model="BAAI/bge-reranker-base",
        documents: Optional[List[Document]] = None,
        verbose=False,
    ):
        index = SentenceWindowRetrievalPipeline.from_default_index(
            llm=llm,
            persist_dir=persist_dir,
            sentence_window_size=sentence_window_size,
            verbose=verbose,
            documents=documents,
            embed_model=embed_model,
        )

        query_engine = SentenceWindowRetrievalPipeline.from_query_engine(
            index=index,
            similarity_top_k=similarity_top_k,
            rerank_top_n=rerank_top_n,
            rerank_model=rerank_model,
            verbose=verbose,
        )

        return query_engine

    def get_llm_with_tools(
        cls, llm=None, documents=None, sentence_window_size=3, *args, **kwargs
    ):
        from llama_index.tools import QueryEngineTool, ToolMetadata

        sentence_window_index = cls.get_sentence_window_index(
            llm=cls.llm,
            persist_dir=cls.persist_dir,
            sentence_window_size=cls.sentence_window_size,
            verbose=cls.verbose,
            documents=documents,
        )

        individual_query_engine_tools = [
            QueryEngineTool(
                query_engine=sentence_window_index.as_query_engine(),
                metadata=ToolMetadata(
                    name=f"ProductSearch",
                    description="useful for when you need to answer questions about the food and drinks menu information",
                ),
            )
        ]

        tools = individual_query_engine_tools  # + [query_engine_tool]
        from llama_index.agent import OpenAIAgent, ReActAgent
        from llama_index.prompts import PromptTemplate
        from llama_index.llms import ChatMessage, MessageRole

        custom_chat_history = [
            ChatMessage(
                role=MessageRole.SYSTEM,
                content="You are working for Delhi Tummy a north indian food delivery company. Your name is Akshat Singh and the conversation will happen over the call.",
            ),
        ]

        # agent = OpenAIAgent.from_tools(tools, verbose=True)
        agent = ReActAgent.from_tools(
            tools, verbose=True, chat_history=custom_chat_history
        )

        return agent

    @staticmethod
    def from_default_index(
        llm=None,
        persist_dir: str = None,
        sentence_window_size=3,
        verbose: bool = False,
        documents: List[Document] = None,
        *args,
        **kwargs,
    ):
        window_metadata_key = kwargs.pop("window_metadata_key", "window")
        original_text_metadata_key = kwargs.pop(
            "original_text_metadata_key", "original_text"
        )
        embed_model = kwargs.pop("embed_model", "local:BAAI/bge-small-en-v1.5")

        node_parser = SentenceWindowNodeParser.from_defaults(
            window_size=sentence_window_size,
            window_metadata_key=window_metadata_key,
            original_text_metadata_key=original_text_metadata_key,
        )

        llm = llm or OpenAI(model="gpt-3.5-turbo")
        sentence_context = ServiceContext.from_defaults(
            llm=llm,
            embed_model=embed_model,
            node_parser=node_parser,
        )

        persist_dir = persist_dir or SENTENCE_WINDOW_INDEX_DEFAULT_DIR

        if not os.path.exists(persist_dir):
            if verbose:
                print(f"Creating sentence index to {persist_dir}")

            documents = SentenceWindowRetrievalPipeline.resolve_documents(documents)

            sentence_index = VectorStoreIndex.from_documents(
                documents, service_context=sentence_context
            )

            sentence_index.storage_context.persist(persist_dir=persist_dir)

        else:
            if verbose:
                print(f"Loading sentence index from {persist_dir}")

            sentence_index = load_index_from_storage(
                StorageContext.from_defaults(persist_dir=persist_dir),
                service_context=sentence_context,
            )

        return sentence_index

    @staticmethod
    def from_query_engine(
        index: VectorStoreIndex,
        similarity_top_k: int = 6,
        rerank_top_n: int = 2,
        rerank_model="BAAI/bge-reranker-base",
        verbose=False,
        *args,
        **kwargs
        # sentence_index, similarity_top_k=6, rerank_top_n=2
    ) -> RetrieverQueryEngine:
        window_metadata_key = kwargs.pop("window_metadata_key", "window")

        # takes value stored in metadata and replaces a node text with that value
        postproc = MetadataReplacementPostProcessor(
            target_metadata_key=window_metadata_key
        )

        # re ranking
        rerank = SentenceTransformerRerank(top_n=rerank_top_n, model=rerank_model)

        sentence_window_engine = index.as_query_engine(
            similarity_top_k=similarity_top_k,  # we need more larger context to be fetched and feed to re rank
            node_postprocessors=[postproc, rerank],
        )

        return sentence_window_engine
