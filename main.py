from trulens_utils.utils import LlamaTrulensEval
import dotenv
from llama_utils.utils import get_single_document
from llama_utils.agent import SentenceWindowRetrievalPipeline
from llama_index.llms import OpenAI


if __name__ == "__main__":
    dotenv.load_dotenv()
    llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)

    document = get_single_document(input_files=["./example/FoodnDrinksCatalogue.txt"])

    pipeline = SentenceWindowRetrievalPipeline(llm=llm, verbose=True)

    sentence_window_engine = pipeline.from_query_engine([document])

    trueval = LlamaTrulensEval(
        query_engine=sentence_window_engine, app_id="Sentence engine 1", fresh=True
    )

    trueval.run_evals(eval_questions=["What are the drinks in the menu ?"])
