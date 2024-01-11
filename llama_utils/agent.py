from llama_index import (
    VectorStoreIndex,
    ServiceContext,
    load_index_from_storage,
    StorageContext,
    Document,
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
from typing import List


class AutoMergingRetrievalPipeline:
    def build_automerging_index(
        documents,
        llm,
        embed_model="local:BAAI/bge-small-en-v1.5",
        save_dir="merging_index",
        chunk_sizes=None,
    ):
        chunk_sizes = chunk_sizes or [2048, 512, 128]
        node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=chunk_sizes)
        nodes = node_parser.get_nodes_from_documents(documents)
        leaf_nodes = get_leaf_nodes(nodes)
        merging_context = ServiceContext.from_defaults(
            llm=llm,
            embed_model=embed_model,
        )
        storage_context = StorageContext.from_defaults()
        storage_context.docstore.add_documents(nodes)

        if not os.path.exists(save_dir):
            automerging_index = VectorStoreIndex(
                leaf_nodes,
                storage_context=storage_context,
                service_context=merging_context,
            )
            automerging_index.storage_context.persist(persist_dir=save_dir)
        else:
            automerging_index = load_index_from_storage(
                StorageContext.from_defaults(persist_dir=save_dir),
                service_context=merging_context,
            )
        return automerging_index

    def get_automerging_query_engine(
        automerging_index,
        similarity_top_k=12,
        rerank_top_n=6,
    ):
        base_retriever = automerging_index.as_retriever(
            similarity_top_k=similarity_top_k
        )
        retriever = AutoMergingRetriever(
            base_retriever, automerging_index.storage_context, verbose=True
        )
        rerank = SentenceTransformerRerank(
            top_n=rerank_top_n, model="BAAI/bge-reranker-base"
        )
        auto_merging_engine = RetrieverQueryEngine.from_args(
            retriever, node_postprocessors=[rerank]
        )
        return auto_merging_engine


class SentenceWindowRetrievalPipeline:
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

    def from_query_engine(cls, documents: List[Document] = None):
        cls.index = cls.build_index(documents)

        cls.query_engine = cls.build_query_engine()

        return cls.query_engine

    def build_index(cls, documents: List[Document] = None, *args, **kwargs):
        window_metadata_key = kwargs.pop("window_metadata_key", "window")
        original_text_metadata_key = kwargs.pop(
            "original_text_metadata_key", "original_text"
        )

        cls.node_parser = SentenceWindowNodeParser.from_defaults(
            window_size=cls.sentence_window_size,
            window_metadata_key=window_metadata_key,
            original_text_metadata_key=original_text_metadata_key,
        )

        sentence_context = ServiceContext.from_defaults(
            llm=cls.llm,
            embed_model=cls.embed_model,
            node_parser=cls.node_parser,
        )

        if not os.path.exists(cls.persist_dir):
            if cls.verbose:
                print(f"Creating sentence index to {cls.persist_dir}")

            sentence_index = VectorStoreIndex.from_documents(
                documents, service_context=sentence_context
            )

            sentence_index.storage_context.persist(persist_dir=cls.persist_dir)

        else:
            if cls.verbose:
                print(f"Loading sentence index from {cls.persist_dir}")

            sentence_index = load_index_from_storage(
                StorageContext.from_defaults(persist_dir=cls.persist_dir),
                service_context=sentence_context,
            )

        cls.sentence_index = sentence_index
        return sentence_index

    def build_query_engine(
        cls,
        **kwargs
        # sentence_index, similarity_top_k=6, rerank_top_n=2
    ) -> RetrieverQueryEngine:
        window_metadata_key = kwargs.pop("window_metadata_key", "window")

        # takes value stored in metadata and replaces a node text with that value
        postproc = MetadataReplacementPostProcessor(
            target_metadata_key=window_metadata_key
        )

        # re ranking
        rerank = SentenceTransformerRerank(
            top_n=cls.rerank_top_n, model=cls.rerank_model
        )

        sentence_window_engine = cls.index.as_query_engine(
            similarity_top_k=cls.similarity_top_k,  # we need more larger context to be fetched and feed to re rank
            node_postprocessors=[postproc, rerank],
        )

        cls.engine = sentence_window_engine
        return cls.engine
