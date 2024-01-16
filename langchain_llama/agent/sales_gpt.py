from langchain.chat_models import ChatOpenAI, ChatLiteLLM
from langchain.chains.base import Chain
from typing import List, Dict, Any, Union, Callable
from pydantic import Field
from langchain_utils.chain.chains import StageAnalyzerChain, ConversationChain, LLMChain
from langchain_utils.utils.constant import Constant as C
from langchain_utils.utils.tools import setup_knowledge_base, get_tools
from langchain_utils.prompt.templates import CustomPromptTemplateForTools
from langchain_utils.utils.parsers import SalesConvoOutputParser
from langchain.agents import LLMSingleActionAgent, AgentExecutor
from langchain_core.language_models.llms import create_base_retry_decorator
from litellm import acompletion
from utils.console import printBold
from langchain_core.messages import SystemMessage
from langchain_utils.prompt.prompts import (
    SALES_AGENT_TOOLS_PROMPT,
    STAGE_ANALYZER_INCEPTION_PROMPT,
    CONVERSATION_AGENT_INCEPTION_PROMPT,
)
from langchain_utils.prompt.variables import (
    CONVERSATION_PROMPT_VARIABLES,
    CONVERSATION_PROMPT_PARTIAL_VARIABLES,
    CONVERSATION_TOOLS_PROMPT_PARTIAL_VARIABLES,
    CONVERSATION_TOOLS_PROMPT_VARIABLES,
    AI_PREFIX,
)
from langchain_utils.utils.stages import (
    CONVERSATION_STAGES,
    END_CONVERSATION_STAGE_ID,
    START_CONVERSATION_STAGE_ID,
)


def _create_retry_decorator(llm: Any) -> Callable[[Any], Any]:
    import openai

    errors = [
        openai.Timeout,
        openai.APIError,
        openai.APIConnectionError,
        openai.RateLimitError,
        openai.APIStatusError,
    ]
    return create_base_retry_decorator(error_types=errors, max_retries=llm.max_retries)


class SalesGPT(Chain):
    # conversation vars
    conversation_history: List[str] = []
    conversation_stage_id: str = START_CONVERSATION_STAGE_ID
    current_conversation_stage: str = CONVERSATION_STAGES.get(
        START_CONVERSATION_STAGE_ID
    )
    conversation_stages: Dict = CONVERSATION_STAGES

    # chains
    stage_analyzer_chain: StageAnalyzerChain = Field(...)
    sales_conversation_utterance_chain: ConversationChain = Field(...)
    sales_agent_executor: Union[AgentExecutor, None] = Field(...)

    # model
    model: str = C.DEFAULT_MODEL_NAME

    # Other vars
    use_tools: bool = False
    verbose: bool = True

    @property
    def input_keys(self) -> List[str]:
        return []

    @property
    def output_keys(self) -> List[str]:
        return []

    def retrieve_conversation_stage(self, key):
        return self.conversation_stages.get(key, START_CONVERSATION_STAGE_ID)

    def seed_agent(self) -> None:
        self.conversation_history = []
        self.current_conversation_stage = self.retrieve_conversation_stage(
            START_CONVERSATION_STAGE_ID
        )

    def determine_conversation_stage(self) -> None:
        self.conversation_stage_id = self.stage_analyzer_chain.run(
            conversation_history="\n".join(self.conversation_history).rstrip("\n"),
            conversation_stage_id=self.conversation_stage_id,
            conversation_stages="\n".join(
                [
                    f"{conv_id} : {conv_val}"
                    for conv_id, conv_val in self.conversation_stages.items()
                ]
            ),
        )

        self.current_conversation_stage = self.retrieve_conversation_stage(
            self.conversation_stage_id
        )

        if self.verbose:
            print(f"Conversation_stage: {self.conversation_stage_id}")
            print(f"Current conv stage: {self.current_conversation_stage}")

    def ai_step(self, ai_input):
        if C.END_OF_TURN not in ai_input:
            ai_input += f" {C.END_OF_TURN}"
        self.conversation_history.append(f"{AI_PREFIX}: {ai_input}")

    def human_step(self, human_input):
        self.conversation_history.append(f"{C.user}: {human_input} {C.END_OF_TURN}")

    def step(self, stream: bool = False):
        if not stream:
            return self._call(inputs={})

        return self._streaming_generator()

    def astep(self, stream: bool = False):
        if not stream:
            return self._acall(inputs={})

        return self._astreaming_generator()

    def _prep_messages(self):
        """
        Helper function to prepare messages to be passed to a streaming generator
        """
        prompt = self.sales_conversation_utterance_chain.prep_prompts(
            [
                dict(
                    conversation_stage=self.current_conversation_stage,
                    conversation_history="\n".join(self.conversation_history),
                    # **self.prompt_data,
                    **CONVERSATION_PROMPT_VARIABLES,
                )
            ]
        )

        inception_messages = prompt[0][0].to_messages()
        message_dict = SystemMessage(content=inception_messages[0].content)

        if self.sales_conversation_utterance_chain.verbose:
            printBold(inception_messages[0].content)
            # print(f"\033[92m{inception_messages[0].content}\033[0m]")

        return [message_dict]

    def _streaming_generator(self):
        messages = self._prep_messages()

        return self.sales_conversation_utterance_chain.llm.completion_with_retry(
            messages=messages, stop=C.END_OF_TURN, stream=True, model=self.model_name
        )

    async def acompletion_with_retry(self, llm: Any, **kwargs: Any) -> Any:
        retry_decorator = _create_retry_decorator(llm)

        @retry_decorator
        async def _completion_with_retry(**kwargs: Any) -> Any:
            return await acompletion(**kwargs)

        return await _completion_with_retry(**kwargs)

    async def _astreaming_generator(self):
        messages = self._prep_messages()

        return await self.acompletion_with_retry(
            llm=self.sales_conversation_utterance_chain.llm,
            messages=messages,
            stop=C.END_OF_TURN,
            stream=True,
            model=self.model_name,
        )

    def _call(self, inputs: Dict[str, Any]) -> str:
        if self.use_tools:
            ai_message = self.sales_agent_executor.run(
                input="",
                conversation_stage=self.current_conversation_stage,
                conversation_history="\n".join(self.conversation_history).strip("\n"),
                **CONVERSATION_TOOLS_PROMPT_VARIABLES,
                conversation_stages="\n".join(
                    [
                        f"{conv_id} : {conv_val}"
                        for conv_id, conv_val in self.conversation_stages.items()
                    ]
                ),
            )
        else:
            ai_message = self.sales_conversation_utterance_chain.run(
                conversation_stage=self.current_conversation_stage,
                conversation_history="\n".join(self.conversation_history).strip("\n"),
                # **self.prompt_data,
                **CONVERSATION_PROMPT_VARIABLES,
                conversation_stages="\n".join(
                    [
                        f"{conv_id} : {conv_val}"
                        for conv_id, conv_val in self.conversation_stages.items()
                    ]
                ),
            )

        # create ai response dialog
        agent_name = AI_PREFIX
        ai_message = f"{agent_name} : {ai_message}"

        # ai message should have end_of_turn token
        if C.END_OF_TURN not in ai_message:
            ai_message += f" {C.END_OF_TURN}"

        self.conversation_history.append(ai_message)

        if self.verbose:
            print(ai_message.replace(C.END_OF_TURN, ""))

        return ai_message.replace(C.END_OF_TURN, "").replace(f"{AI_PREFIX} :", "")


    def _acall(self, *args, **kwargs):
        raise NotImplementedError("This method has not been implemented yet.")

    @classmethod
    def from_llm(
        self,
        llm: Union[ChatOpenAI, ChatLiteLLM] = None,
        retriever_type="llamaindex-sentence_window",  # lanchain, llamaindex-sentence_window
        verbose: bool = True,
        use_tools: bool = False,
        **kwargs,
    ) -> "SalesGPT":
        """
        Create a prompt template for all the chains as user will send the partial data eg. stages
        and all the prompt will be designed by us.
        """

        llm = llm or ChatOpenAI(model_name = C.DEFAULT_MODEL_NAME)
        stage_analyzer_chain = StageAnalyzerChain.from_llm(
            llm, verbose, prompt=STAGE_ANALYZER_INCEPTION_PROMPT
        )

        sales_conversation_utterance_chain = ConversationChain.from_llm(
            llm=llm,
            verbose=verbose,
            prompt_variables=[
                *CONVERSATION_PROMPT_PARTIAL_VARIABLES,
                *CONVERSATION_PROMPT_VARIABLES,
            ],
            prompt=CONVERSATION_AGENT_INCEPTION_PROMPT,
        )

        if use_tools:

            if retriever_type.startswith("llamaindex"):
                from llama_index.langchain_helpers.agents import (
                    IndexToolConfig,
                    LlamaIndexTool,
                )

                from llama_utils.utils.retrievers import (
                    SentenceWindowRetrievalPipeline,
                    AutoMergingRetrievalPipeline,
                )
                from llama_utils.utils.helper import get_single_document

                # query_engine = SentenceWindowRetrievalPipeline.from_default_query_engine()
                retrieval_pipeline = (
                    SentenceWindowRetrievalPipeline
                    if retriever_type.split("-")[1] == "sentence_window"
                    else AutoMergingRetrievalPipeline
                )

                documents = kwargs.get(C.DOCUMENTS_KWARG, [])

                query_engine = retrieval_pipeline.from_default_query_engine(
                    documents=[get_single_document(documents)]
                )

                tool_config = IndexToolConfig(
                    query_engine=query_engine,
                    name=f"SearchMenu",
                    description=f"use this tool when you need to search things related to the menu of the restraunt.",
                    tool_kwargs={"return_direct": True},
                )
                tools = LlamaIndexTool.from_tool_config(tool_config)
                tools = [tools]
                knowledge_base = None

            elif retriever_type == "langchain":
                product_catalog = documents = kwargs.get("documents", [])
                knowledge_base = setup_knowledge_base(product_catalog)
                tools = get_tools(knowledge_base)

            prompt = CustomPromptTemplateForTools(
                template=SALES_AGENT_TOOLS_PROMPT,
                tools_getter=lambda x: tools,
                input_variables=[
                    *CONVERSATION_TOOLS_PROMPT_VARIABLES,
                    *CONVERSATION_TOOLS_PROMPT_PARTIAL_VARIABLES,
                ],
            )

            llm_chain = LLMChain(llm=llm, prompt=prompt, verbose=verbose)
            tool_names = [tool.name for tool in tools]

            output_parser = SalesConvoOutputParser(ai_prefix=AI_PREFIX)

            sales_agent_with_tools = LLMSingleActionAgent(
                llm_chain=llm_chain,
                output_parser=output_parser,
                stop=[C.OBSERVATION],
                allowed_tools=tool_names,
            )

            sales_agent_executor = AgentExecutor.from_agent_and_tools(
                agent=sales_agent_with_tools,
                tools=tools,
                verbose=verbose,
            )
        else:
            sales_agent_executor = None
            knowledge_base = None

        model_name = ""
        if isinstance(llm, ChatLiteLLM):
            model_name = llm.model
        if isinstance(llm, ChatOpenAI):
            model_name = llm.model_name

        return self(
            # prompt_data=prompt_data,
            stage_analyzer_chain=stage_analyzer_chain,
            sales_conversation_utterance_chain=sales_conversation_utterance_chain,
            sales_agent_executor=sales_agent_executor,
            knowledge_base=knowledge_base,
            model_name=model_name,
            verbose=verbose,
            use_tools=use_tools,
            **kwargs,
        )
