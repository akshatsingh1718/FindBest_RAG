from langchain.chains import LLMChain
from langchain.chat_models import ChatLiteLLM, ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_utils.prompt.prompts import CONVERSATION_AGENT_INCEPTION_PROMPT, STAGE_ANALYZER_INCEPTION_PROMPT
from langchain_utils.utils.dummy_details import DummyDet1 as example
from typing import Union, List


class ConversationChain(LLMChain):
    """Chain to generate the next utterance for the conversation

    In the prompt given the message history was passed.
    """
    custom_prompt_variables: List[str] = []

    @classmethod
    def from_llm(
        self,
        llm: Union[ChatLiteLLM, ChatOpenAI],
        verbose: bool = True,
        prompt_variables: List[str] = None,
        prompt : str = ""
    ):
        prompt = PromptTemplate(
            template= prompt,
            input_variables=[
                *prompt_variables,
                "conversation_history",
            ],
        )

        # we return the LLMChain instance
        return self(prompt=prompt, llm=llm, verbose=verbose)


class StageAnalyzerChain(LLMChain):
    @classmethod
    def from_llm(
        cls, 
        llm: Union[ChatLiteLLM, ChatOpenAI], 
        verbose: bool = True,
        prompt: str= None,
    ) -> LLMChain:
        prompt = PromptTemplate(
            template= prompt,
            input_variables=[
                "conversation_history",
                "conversation_stage_id",
                "conversation_stages",
            ],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)


def conversation_chain_example():
    llm = ChatOpenAI()

    # here we have LLMChain instance
    chain = ConversationChain.from_llm(
        llm=llm,
        verbose=False,
    )

    def complete(history: list):
        return chain.run(
            conversation_history="\n".join(history),
            salesperson_name="Akshat Singh",
            salesperson_role="Phone order operator",
            company_name="Delhi Tummy",
            company_business="""Delhi Tummy is a food delivery company which makes delicious north indian food and street food. Our food recipes are created by some of the finest food chefs in Delhi.""",
            company_values="Out mission at Delhi Tummy is to provide good and healthy food in a very affordable price. We want everyone single indian to eat food without any hesitation of price and quality of the food.",
            conversation_purpose="Give them a very healthy conversation so that they can find out the best food for the day.",
            conversation_type="call",
        )

    def format_ai_resp(ai_msg):
        ai_msg = agent_name + ": " + ai_msg
        print(ai_msg)
        if "<END_OF_TURN>" not in ai_msg:
            ai_msg += " <END_OF_TURN>"
        return ai_msg

    # create empty conversation history
    conversation_history = []

    human_input = ""
    agent_name = example.salesperson_name

    # Greetings from AI
    ai_message = complete(conversation_history)

    ai_message = format_ai_resp(ai_msg=ai_message)

    conversation_history.append(ai_message)

    while True:
        human_input = input("User: ")

        if human_input.lower()[0] == "q":
            break
        if human_input.lower() == "c":
            print("*" * 40 + "chains.py")
            print("\n".join(conversation_history))
            print("*" * 40)
            continue

        conversation_history.append(f"User: {human_input} <END_OF_TURN>")

        # Here we will run the chain which needs input variables for prompt template
        ai_message = complete(conversation_history)

        ai_message = format_ai_resp(ai_msg=ai_message)

        conversation_history.append(ai_message)


def stage_analyzer_chain_example():
    CONVERSATION_STAGES= {
        "1": "Introduction: Start the conversation by introducing yourself and your company. Be polite and respectful while keeping the tone of the conversation professional. Your greeting should be welcoming. Always clarify in your greeting the reason why you are calling.", 
        "2": "Qualification: Qualify the prospect by confirming if they are the right person to talk to regarding your product/service. Ensure that they have the authority to make purchasing decisions.",
        "3": "Value proposition: Briefly explain how your product/service can benefit the prospect. Focus on the unique selling points and value proposition of your product/service that sets it apart from competitors.",
        "4": "Needs analysis: Ask open-ended questions to uncover the prospect's needs and pain points. Listen carefully to their responses and take notes.",
        "5": "Solution presentation: Based on the prospect's needs, present your product/service as the solution that can address their pain points.",
        "6": "Objection handling: Address any objections that the prospect may have regarding your product/service. Be prepared to provide evidence or testimonials to support your claims.",
        "7": "Close: Ask for the sale by proposing a next step. This could be a demo, a trial or a meeting with decision-makers. Ensure to summarize what has been discussed and reiterate the benefits.",
        "8": "End conversation: It's time to end the call as there is nothing else to be said."
    }
    llm = ChatOpenAI()

    chain = StageAnalyzerChain.from_llm(llm=llm, verbose=True)

    conversation_history = [f"{example.salesperson_name} : Hello how are you.", "User: Who is this ?"]
    current_conversation_stage_id = "1"

    next_stage_id = chain.run(
        conversation_history="\n".join(conversation_history).rstrip("\n"),
        conversation_stage_id=current_conversation_stage_id,
        conversation_stages="\n".join(
            [
                f"{stage_id} : {stage_desc}"
                for stage_id, stage_desc in CONVERSATION_STAGES.items()
            ]
        ),
    )

    print(f"{next_stage_id = }")
    print(CONVERSATION_STAGES[next_stage_id])

if __name__ == "__main__":
    import os
    import dotenv
    import openai

    dotenv.load_dotenv("../../.env")
    openai.api_key = os.environ.get("OPENAI_API_KEY")

    stage_analyzer_chain_example()
