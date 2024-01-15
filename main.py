from trulens_utils.utils import LlamaTrulensEval
import dotenv
from llama_utils.utils.helper import get_single_document
from llama_utils.utils.retrievers import (
    SentenceWindowRetrievalPipeline,
    AutoMergingRetrievalPipeline,
)
from llama_index.llms import OpenAI
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


def check_retrievers():
    llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)

    document = get_single_document(input_files=["./example/FoodnDrinksCatalogue.txt"])

    window_pipeline = SentenceWindowRetrievalPipeline(llm=llm, verbose=True)
    automerging_pipeline = AutoMergingRetrievalPipeline(llm=llm, verbose=True)

    sentence_window_engine = window_pipeline.from_query_engine([document])
    automerging_engine = automerging_pipeline.from_query_engine([document])

    window_trueval = LlamaTrulensEval(
        query_engine=sentence_window_engine, app_id="Sentence engine 1", fresh=True
    )
    auto_trueval = LlamaTrulensEval(
        query_engine=automerging_engine, app_id="Automerging engine 1", fresh=True
    )

    auto_trueval.run_evals(eval_questions=["What are the drinks in the menu ?"])
    window_trueval.run_evals(eval_questions=["What are the drinks in the menu ?"])


def check_llmWithTool():
    llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)

    pipeline = SentenceWindowRetrievalPipeline(llm=llm, verbose=True)
    agent = pipeline.get_llm_with_tools(llm=llm)

    while True:
        query = input("User :")
        print(agent.chat(query))


def check_llama_agent():
    llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
    from llama_utils.agents.LlamaGPT import LlamaGPTAgent

    agent = LlamaGPTAgent.from_llm(
        llm,
        documents=["./example/FoodnDrinksCatalogue.txt"],
        sys_msg="You are working for Delhi Tummy a north indian food delivery company. Your name is Akshat Singh and the conversation will happen over the call.",
        verbose= True,
        retriever="sentence-window",
    )

    ai_greetings= "How are you today? I hope you're having a great day so far. My name is Akshat Singh and I'm calling from Delhi Tummy. We are a food delivery company that specializes in delicious North Indian food and street food. How can I assist you today?"
    agent.ai_step(ai_greetings) 

    while True:
        query = input("User: ")
        agent.human_step( query )

if __name__ == "__main__":
    dotenv.load_dotenv()

    check_llama_agent()
