from .utils import LlamaIndexMultiRetrieverEval
from datetime import datetime
import sys

def main():
    EVAL_QUESTIONS = "example/questions.txt"
    DOCUMENTS = "./example/FoodnDrinksCatalogue.txt"
    eval_ques = []
    with open(EVAL_QUESTIONS, "r") as file:
        eval_ques = file.readlines()

    eval_ques = eval_ques[:1]
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    db_file_name = f"db_{current_datetime}.sqlite"

    eval = LlamaIndexMultiRetrieverEval(
        eval_questions=eval_ques,
        database_file=f"./data/trulens_db/{db_file_name}",
        fresh=True,
        documents= DOCUMENTS,
    )

    eval.find_best_retriever()



if __name__ == "__main__":
    main()

    sys.exit()
