from agent.agent import SalesGPT
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import json
import argparse
from prompt.variables import AI_PREFIX
from conversation.stages import END_CONVERSATION_STAGE_ID

class ChatEngine:
    def __init__(
        self,
        llm,
        catalogue_path: str,
        verbose: bool = False,
        use_tools: bool = False,
    ):
        self.llm = llm
        self.catalogue_path = catalogue_path
        self.verbose = verbose
        self.use_tools = use_tools

    def json_to_dict(self, path: str) -> dict:
        data = {}
        with open(path) as f:
            data = json.load(f)

        return data

    def start(self):
        print("****************** Start")
        agent = SalesGPT.from_llm(
            llm=self.llm,
            use_tools=self.use_tools,
            verbose=self.verbose,
            product_catalog=self.catalogue_path,
        )

        agent.seed_agent()
        # agent.determine_conversation_stage()
        # agent
        # agent.step()

        ai_greetings= "How are you today? I hope you're having a great day so far. My name is Akshat Singh and I'm calling from Delhi Tummy. We are a food delivery company that specializes in delicious North Indian food and street food. How can I assist you today?"
        agent.ai_step(ai_greetings) 
        print(f"{AI_PREFIX}: {ai_greetings}")

        while True:
            # user
            user_input = input("User: ")

            if user_input.lower() == "q":
                break

            agent.human_step(user_input)

            # agent
            # agent.determine_conversation_stage()
            agent.step()

            # print("=" * 40)
            # print(agent.conversation_stage_id)
            # print("=" * 40)

            if agent.conversation_stage_id == END_CONVERSATION_STAGE_ID:
                break


if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--prompt_data_path", help="Prompt data json path"
    )
    parser.add_argument("-s", "--stages_path", help="Stages json path eg. stages.json")
    parser.add_argument("-c", "--catalogue_path", help="Company catalogue details path eg catalogue.txt")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose")
    parser.add_argument(
        "-t", "--usetools", action="store_true", help="Add This when to use tools"
    )

    # Read arguments from command line
    args = parser.parse_args()

    engine = ChatEngine(
        llm=ChatOpenAI(temperature=0.4, model_name="gpt-3.5-turbo", streaming=True),
        catalogue_path=args.catalogue_path,
        verbose=args.verbose,
        use_tools=args.usetools,
    )

    engine.start()
