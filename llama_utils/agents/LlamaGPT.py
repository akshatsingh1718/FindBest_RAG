from llama_index import (
    Document,
)
from llama_index.llms import OpenAI
from llama_utils.utils.retrievers import (
    AutoMergingRetrievalPipeline,
    SentenceWindowRetrievalPipeline,
)
from llama_utils.utils.helper import get_single_document
from typing import Dict, Any, Optional, List, Union
from llama_index.tools import QueryEngineTool, ToolMetadata
from llama_index.agent import OpenAIAgent
from llama_utils.utils.common import (
    AUTO_MERGING_INDEX,
    SENTENCE_WINDOW_INDEX,
    OPENAI_DEFAULT_MODEL,
)

"""
https://docs.llamaindex.ai/en/stable/examples/low_level/vector_store.html
"""


class LlamaGPTAgent:
    def __init__(cls, llm, conversation_history, query_engine, sys_msg, verbose=False):
        cls.llm = llm
        cls.verbose = verbose
        cls.conversation_history = conversation_history
        cls.query_engine = query_engine
        cls.sys_msg = sys_msg

        cls.tools = cls._get_tools(query_engine)
        cls.agent_with_tools = OpenAIAgent.from_tools(
            cls.tools, llm=llm, verbose=True, chat_history=[], system_prompt=sys_msg
        )

    @classmethod
    def from_llm(
        cls,
        llm=None,
        sys_msg: str = None,
        verbose: bool = True,
        use_tools: bool = True,
        documents: Optional[Union[List[str], Document, List[Document]]] = [],
        retriever: Optional[str] = "auto-merging",
        *retriever_args,
        **retriever_kwargs,
    ) -> "LlamaGPTAgent":
        llm = llm or OpenAI(model=OPENAI_DEFAULT_MODEL)
        if not isinstance(llm, OpenAI):
            raise NotImplementedError("Only OpenAI models can be used for this")
        cls.use_tools = use_tools

        if isinstance(documents, Document):
            documents = [documents]
        elif all(isinstance(doc, str) for doc in documents):
            documents = [get_single_document(input_files=documents)]
        elif all(isinstance(doc, Document) for doc in documents):
            documents = Document(text="\n\n".join([doc.text for doc in documents]))
        else:
            raise TypeError(f"Given documents type are not supported.")

        print(retriever)
        if retriever == SENTENCE_WINDOW_INDEX:
            print("-------> Sentence  Window")
            query_engine = SentenceWindowRetrievalPipeline.from_default_query_engine(
                llm=llm,
                documents=documents,
                verbose=verbose,
                *retriever_args,
                **retriever_kwargs,
            )

        elif retriever == AUTO_MERGING_INDEX:
            print("-------> Auto Merging")
            query_engine = AutoMergingRetrievalPipeline.from_default_query_engine(
                llm=llm,
                documents=documents,
                verbose=verbose,
                *retriever_args,
                **retriever_kwargs,
            )

        return cls(
            llm=llm,
            conversation_history=[],
            query_engine=query_engine,
            sys_msg=sys_msg,
            verbose=verbose,
        )

    def ai_step(cls, ai_input: str):
        # cls.memory.add_message("assistant", ai_input)
        cls.conversation_history.append(ai_input)
        from llama_index.llms import ChatMessage

        cls.agent_with_tools.chat_history.append(
            ChatMessage(role="assistant", content=ai_input)
        )

    def human_step(cls, human_input: str):
        ai_message = cls.agent_with_tools.chat(human_input)

        if cls.verbose:
            print("AI: ", ai_message)
        return ai_message

    @staticmethod
    def _get_tools(query_engine) -> List[QueryEngineTool]:
        query_engine_tools = [
            QueryEngineTool(
                query_engine=query_engine,
                metadata=ToolMetadata(
                    name=f"FoodMenuSearch",
                    description="useful for when you need to answer questions about the food and drinks menu information",
                ),
            )
        ]

        return query_engine_tools

    def step(cls, stream: bool = False):
        if not stream:
            return cls._call(inputs={})

        return cls._streaming_generator()
