from itertools import product
from llama_utils.utils.retrievers import SentenceWindowRetrievalPipeline
from trulens_utils.evaluations import LlamaindexTrulensEval
from trulens_utils.utils.evals_questions import EVAL_QUESTION
from llama_index.llms import OpenAI
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

llm = OpenAI(model="gpt-3.5-turbo")

embed_model = "local:BAAI/bge-small-en-v1.5"
rerank_model = "BAAI/bge-reranker-base"
persist_dir = ""
documents = [None]

sentence_window_size_list = [1, 2, 3, 4, 5, 6]
similarity_top_k_list = [2, 4, 6, 8, 10]
rerank_top_n_list = [2, 4, 6]


parameter_combinations = product(
    sentence_window_size_list, similarity_top_k_list, rerank_top_n_list
)

DATABASE_FILE = "trulens_utils/llama_index/grid_search_results.sqlite"


for sentence_window_size, similarity_top_k, rerank_top_n in parameter_combinations:
    print("\n", "=" * 50)
    app_id = f"win={sentence_window_size}; top_k= {similarity_top_k}; r_top_n= {rerank_top_n}"

    print(f"{app_id}")
    retriever = SentenceWindowRetrievalPipeline.from_default_query_engine(
        llm=llm,
        embed_model=embed_model,
        sentence_window_size=sentence_window_size,
        persist_dir="",
        similarity_top_k=similarity_top_k,
        rerank_top_n=rerank_top_n,
        rerank_model=rerank_model,
        documents=documents,
    )

    trulens = LlamaindexTrulensEval(
        query_engine=retriever, app_id=app_id, database_file=DATABASE_FILE
    )

    trulens.run_evals(eval_questions=EVAL_QUESTION, show_progress=True, start_dashboard=True)

    trulens.show_leaderboard()
    print("=" * 50)
