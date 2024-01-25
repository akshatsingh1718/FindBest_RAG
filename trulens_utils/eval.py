from itertools import product
from llama_utils.utils.retrievers import SentenceWindowRetrievalPipeline
from trulens_utils.utils import LlamaindexTrulensEval
from trulens_utils.evals_questions import EVAL_QUESTION
from llama_index.llms import OpenAI
from tqdm import tqdm
import warnings
import psutil
import gc

warnings.filterwarnings("ignore", category=DeprecationWarning)

llm = OpenAI(model="gpt-3.5-turbo")
DATABASE_FILE = "trulens_utils/llama_index/grid_search_results.sqlite"


def start_eval(app_id, sentence_window_size, similarity_top_k, rerank_top_n, idx):
    at_starting_memory = psutil.virtual_memory().percent
    print(f"[{idx}] Starting {app_id} memory = {at_starting_memory}")

    # index = SentenceWindowRetrievalPipeline.from_default_index(
    #     persist_dir="",
    #     sentence_window_size=sentence_window_size,
    #     documents=["./example/FoodnDrinksCatalogue.txt"],
    # )
    # after_index_memory = psutil.virtual_memory().percent
    # print(
    #     f"After creating index= {after_index_memory} | inc by= {after_index_memory - at_starting_memory}"
    # )

    # retriever = SentenceWindowRetrievalPipeline.from_query_engine(
    #     index,
    #     similarity_top_k=similarity_top_k,
    #     rerank_top_n=rerank_top_n,
    #     rerank_model="BAAI/bge-reranker-base",
    # )
    # after_retriever_memory = psutil.virtual_memory().percent
    # print(
    #     f"After creating retriever= {after_retriever_memory} | inc by= {after_retriever_memory - after_index_memory}"
    # )

    retriever = SentenceWindowRetrievalPipeline.from_default_query_engine(
        llm=llm,
        embed_model="local:BAAI/bge-small-en-v1.5",
        sentence_window_size=sentence_window_size,
        persist_dir="",
        similarity_top_k=similarity_top_k,
        rerank_top_n=rerank_top_n,
        rerank_model="BAAI/bge-reranker-base",
        documents=["./example/FoodnDrinksCatalogue.txt"],
    )
    after_retriever_memory = psutil.virtual_memory().percent
    print(
        f"After creating retriever= {after_retriever_memory} | inc by= {after_retriever_memory - at_starting_memory}"
    )

    trulens = LlamaindexTrulensEval(
        query_engine=retriever, app_id=app_id, database_file=DATABASE_FILE
    )
    after_trulens_memory = psutil.virtual_memory().percent
    print(
        f"After creating trulens= {after_trulens_memory} | inc by= {after_trulens_memory - after_retriever_memory}"
    )

    trulens.run_evals(
        eval_questions=EVAL_QUESTION, show_progress=True, start_dashboard=False
    )
    # trulens.show_leaderboard()

    after_trulens_eval_memory = psutil.virtual_memory().percent
    print(
        f"After trulens eval= {after_trulens_eval_memory} | inc by= {after_trulens_eval_memory - after_trulens_memory}"
    )

    # Even after deleting retriver memory is not freed
    del (
        retriever._node_postprocessors
    )  # deleting re rank model which takes a hefty chunk of memory
    gc.collect()

    after_del_memory = psutil.virtual_memory().percent
    print(
        f"After del= {after_del_memory} | inc/dec by= {after_del_memory - after_trulens_eval_memory}"
    )
    return


def main():
    sentence_window_size_list = [1, 2, 4]
    similarity_top_k_list = [6, 8]
    rerank_top_n_list = [2, 4]

    parameter_combinations = product(
        sentence_window_size_list, similarity_top_k_list, rerank_top_n_list
    )

    for idx, (sentence_window_size, similarity_top_k, rerank_top_n) in enumerate(
        parameter_combinations
    ):
        print("\n\n", "=" * 50)
        app_id = f"win={sentence_window_size}; top_k= {similarity_top_k}; r_top_n= {rerank_top_n}"

        print(f"{app_id}")
        start_eval(app_id, sentence_window_size, similarity_top_k, rerank_top_n, idx)
        print("=" * 50)
        print("=" * 50)


if __name__ == "__main__":
    main()
