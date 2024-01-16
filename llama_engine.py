# from agent.agent import SalesGPT
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import json
import argparse

# # from prompt.variables import AI_PREFIX
# from langchain_llama.utils.stages import END_CONVERSATION_STAGE_ID
# # from langchain_llama.agent.agent import SalesGPT
# from langchain_llama.agent.sales_gpt import SalesGPT
from langchain_llama.prompt.variables import AI_PREFIX
from llama_utils.agents.LlamaGPT import LlamaGPTAgent


class LlamaIndexCLIChatEngine:
    def __init__(
        self,
        llm,
        documents: str,
        verbose: bool = False,
        use_tools: bool = False,
    ):
        self.llm = llm
        self.documents = documents
        self.verbose = verbose
        self.use_tools = use_tools

    def start(self):
        print("****************** Start")
        # agent = SalesGPT.from_llm(
        #     llm=self.llm,
        #     use_tools=self.use_tools,
        #     verbose=self.verbose,
        #     documents=self.documents,
        # )

        agent = LlamaGPTAgent.from_llm(
            documents=["./example/FoodnDrinksCatalogue.txt"],
            sys_msg="You are working for Delhi Tummy a north indian food delivery company. Your name is Akshat Singh and the conversation will happen over the call.",
            verbose=True,
            retriever="sentence-window",
        )

        ai_greetings = "How are you today? I hope you're having a great day so far. My name is Akshat Singh and I'm calling from Delhi Tummy. We are a food delivery company that specializes in delicious North Indian food and street food. How can I assist you today?"
        agent.ai_step(ai_greetings)
        print(f"{AI_PREFIX}: {ai_greetings}")

        while True:
            # user
            user_input = input("User: ")

            if user_input.lower() == "q":
                break

            agent.human_step(user_input)


if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--prompt_data_path", help="Prompt data json path")
    parser.add_argument("-s", "--stages_path", help="Stages json path eg. stages.json")
    parser.add_argument("-d", "--documents", help="documents for RAG")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose")
    parser.add_argument(
        "-t", "--usetools", action="store_true", help="Add This when to use tools"
    )

    # Read arguments from command line
    args = parser.parse_args()

    engine = LlamaIndexCLIChatEngine(
        llm=ChatOpenAI(temperature=0.4, model_name="gpt-3.5-turbo", streaming=True),
        documents=args.documents,
        verbose=args.verbose,
        use_tools=args.usetools,
    )

    engine.start()
