from trulens_eval import Feedback, TruLlama, OpenAI, Tru
from trulens_eval.feedback import Groundedness
import numpy as np
from typing import List
from trulens_utils.constant import Constant as C
from llama_index import load_index_from_storage, StorageContext
from langchain_llama.prompt.variables import (
    CONVERSATION_PROMPT_VARIABLES,
    CONVERSATION_PROMPT_PARTIAL_VARIABLES,
    CONVERSATION_TOOLS_PROMPT_PARTIAL_VARIABLES,
    CONVERSATION_TOOLS_PROMPT_VARIABLES,
    AI_PREFIX,
)
import warnings
from itertools import product
from tqdm import tqdm
from llama_utils.utils.retrievers import (
    BaseRetrievalPipeline,
    SentenceWindowRetrievalPipeline,
    AutoMergingRetrievalPipeline,
)
from llama_index.indices.postprocessor import (
    MetadataReplacementPostProcessor,
    SentenceTransformerRerank,
)
import psutil
import json

warnings.filterwarnings("ignore", category=DeprecationWarning)


class LangchainTrulensEval:
    def __init__(
        cls,
        rag_chain,
        app_id: str,
        database_file: str,
        verbose=False,
        fresh=False,
    ):
        cls.rag_chain = rag_chain
        cls.app_id = app_id
        cls.verbose = verbose
        cls.fresh = fresh
        cls.database_file = database_file

        if cls.fresh:
            if cls.verbose:
                print(f"Resetting database file: {database_file}")
            Tru(database_file=cls.database_file).reset_database()

        cls.trulens_recorder = cls.get_triad_trulens_recorder()

    def get_triad_trulens_recorder(cls):
        # lazy import
        from trulens_eval.app import App
        from trulens_eval import TruChain

        context = App.select_context(cls.rag_chain)

        grounded = Groundedness(groundedness_provider=cls.provider)
        # Define a groundedness feedback function
        f_groundedness = (
            Feedback(grounded.groundedness_measure_with_cot_reasons)
            .on(context.collect())  # collect context chunks into a list
            .on_output()
            .aggregate(grounded.grounded_statements_aggregator)
        )

        # Question/answer relevance between overall question and answer.
        f_qa_relevance = Feedback(
            # cls.provider.relevance, name="Answer Relevance"
            cls.provider.relevance_with_cot_reasons,
            name="Answer Relevance",
        ).on_input_output()
        # Question/statement relevance between question and each context chunk.
        f_context_relevance = (
            # Feedback(cls.provider.qs_relevance, name="Context Relevance")
            Feedback(cls.provider.relevance_with_cot_reasons, name="Context Relevance")
            .on_input()
            .on(context)
            .aggregate(np.mean)
        )
        tru_recorder = TruChain(
            cls.rag_chain,
            app_id="Chain1_ChatApplication",
            feedbacks=[f_qa_relevance, f_context_relevance, f_groundedness],
        )

        return tru_recorder

    def run_evals(cls, eval_questions: List[str], start_dashboard=False):
        if start_dashboard:
            Tru(database_file=cls.database_file).run_dashboard()

        for question in eval_questions:
            with cls.trulens_recorder as recording:
                # response = cls.rag_chain.invoke(question)

                response = chain.rag_chain.invoke(
                    dict(
                        input="",
                        conversation_stage="1",
                        conversation_history=f"user: {question}",
                        **CONVERSATION_TOOLS_PROMPT_VARIABLES,
                        conversation_stages="\n",
                    )
                )
                if cls.verbose:
                    print(response)


class LlamaIndexSingleRetrieverEval:
    def __init__(
        cls,
        query_engine=None,
        app_id="",
        database_file: str = "",
        provider=None,  # open ai
        verbose=False,
        fresh=False,
    ):
        cls.query_engine = query_engine
        cls.app_id = app_id
        cls.verbose = verbose
        cls.fresh = fresh
        cls.database_file = database_file
        cls.provider = provider or OpenAI(model_engine=C.DEFAULT_PROVIDER_MODEL)

        if cls.fresh:
            if cls.verbose:
                print(f"Resetting database file: {database_file}")
            cls.tru = Tru(database_file=cls.database_file).reset_database()

        else:
            cls.tru = Tru(database_file=cls.database_file)

        cls.trulens_recorder = cls.get_triad_trulens_recorder()

    def get_triad_trulens_recorder(cls):
        qa_relevance = Feedback(
            cls.provider.relevance_with_cot_reasons, name="Answer Relevance"
        ).on_input_output()

        qs_relevance = (
            Feedback(cls.provider.relevance_with_cot_reasons, name="Context Relevance")
            .on_input()
            .on(TruLlama.select_source_nodes().node.text)
            .aggregate(np.mean)
        )

        grounded = Groundedness(groundedness_provider=cls.provider)

        groundedness = (
            Feedback(
                grounded.groundedness_measure_with_cot_reasons, name="Groundedness"
            )
            .on(TruLlama.select_source_nodes().node.text)
            .on_output()
            .aggregate(grounded.grounded_statements_aggregator)
        )

        feedbacks = [qa_relevance, qs_relevance, groundedness]
        tru_recorder = TruLlama(
            cls.query_engine, app_id=cls.app_id, feedbacks=feedbacks
        )
        return tru_recorder

    def show_leaderboard(cls):
        leaderboard = cls.tru.get_leaderboard(app_ids=[cls.app_id])
        print(leaderboard)

    def run_evals(
        cls, eval_questions: List[str], start_dashboard=False, show_progress=False
    ):
        if start_dashboard:
            Tru(database_file=cls.database_file).run_dashboard()

        from tqdm import tqdm

        pbar = eval_questions
        if show_progress:
            pbar = tqdm(eval_questions)

        for question in pbar:
            if show_progress:
                pbar.set_description(f"Eval Que= {question}")
            with cls.trulens_recorder as recording:
                response = cls.query_engine.query(question)
                if cls.verbose:
                    print(response)


class LlamaIndexMultiRetrieverEval:
    def __init__(
        cls,
        eval_questions: List[str],
        documents,
        # storage_persist_dir: str,
        database_file: str,
        provider=None,  # open ai
        verbose=False,
        fresh=False,
    ):
        cls.verbose = verbose
        cls.fresh = fresh
        cls.database_file = database_file
        cls.documents = documents
        # cls.storage_persist_dir = storage_persist_dir
        cls.eval_questions = eval_questions
        cls.provider = provider or OpenAI(model_engine=C.DEFAULT_PROVIDER_MODEL)
        cls.config = None
        with open("trulens_utils/config.json", "r") as json_file:
            cls.config = json.load(json_file)

        if cls.fresh:
            if cls.verbose:
                print(f"Resetting database file: {database_file}")
            cls.tru = Tru(database_file=cls.database_file).reset_database()

        else:
            cls.tru = Tru(database_file=cls.database_file)

    def get_feedbacks(cls) -> list:
        qa_relevance = Feedback(
            cls.provider.relevance_with_cot_reasons, name="Answer Relevance"
        ).on_input_output()

        qs_relevance = (
            Feedback(cls.provider.relevance_with_cot_reasons, name="Context Relevance")
            .on_input()
            .on(TruLlama.select_source_nodes().node.text)
            .aggregate(np.mean)
        )

        grounded = Groundedness(groundedness_provider=cls.provider)

        groundedness = (
            Feedback(
                grounded.groundedness_measure_with_cot_reasons, name="Groundedness"
            )
            .on(TruLlama.select_source_nodes().node.text)
            .on_output()
            .aggregate(grounded.grounded_statements_aggregator)
        )

        feedbacks = [qa_relevance, qs_relevance, groundedness]

        return feedbacks

    def find_best_retriever(cls, run_dashboard=True):
        if run_dashboard:
            Tru(database_file=cls.database_file).run_dashboard()

        print("=" * 50)
        print(" " * 20 + " Sentence window retriever")

        sentence_window_size_list = cls.config['SentenceWindowRetrievalPipeline']["sentence_window_size_list"]  # [1, 2, 4]
        similarity_top_k_list = cls.config['SentenceWindowRetrievalPipeline']["similarity_top_k_list"]  # [6]  # [6, 8]
        rerank_top_n_list = cls.config['SentenceWindowRetrievalPipeline']["rerank_top_n_list"]  # [2]  # [2, 4]

        sentence_parameter_combinations = list(
            product(sentence_window_size_list, similarity_top_k_list, rerank_top_n_list)
        )
        for idx, (
            sentence_window_size,
            similarity_top_k,
            rerank_top_n,
        ) in enumerate(sentence_parameter_combinations):
            print("=" * 30)
            print(f"==> Eval at: {idx + 1}/{len(sentence_parameter_combinations)} ")
            print(f"Memory usage: { psutil.virtual_memory().percent}")
            app_id = f"App-SentenceWindow-winSize={sentence_window_size}-simTopK={similarity_top_k}-topN={rerank_top_n}"

            query_engine = SentenceWindowRetrievalPipeline.from_default_query_engine(
                sentence_window_size=sentence_window_size,
                similarity_top_k=similarity_top_k,
                rerank_top_n=rerank_top_n,
                documents=cls.documents,
                persist_dir=f"./data/indexes/{app_id}",
            )

            tru_recorder = TruLlama(
                query_engine, app_id=app_id, feedbacks=cls.get_feedbacks()
            )

            for eval_que in tqdm(cls.eval_questions):
                with tru_recorder as recorder:
                    query_engine.query(eval_que)
            del query_engine._node_postprocessors
            print("=" * 30)


        print("=" * 50)
        print(" " * 20 + " Auto merging retriever")

        rerank_top_n_list = cls.config['AutoMergingRetrievalPipeline']["rerank_top_n_list"]  # [1, 2, 4]
        similarity_top_k_list = cls.config['AutoMergingRetrievalPipeline']["similarity_top_k_list"]  # [6]  # [6, 8]
        check_sizes_list = cls.config['AutoMergingRetrievalPipeline'][
            "check_sizes_list"
        ]  # [[2048, 512, 128], [512, 128], [2048, 512] ]

        automerging_parameter_combinations = list(
            product(check_sizes_list, similarity_top_k_list, rerank_top_n_list)
        )
        for idx, (
            check_sizes,
            similarity_top_k,
            rerank_top_n,
        ) in enumerate(automerging_parameter_combinations):
            print("=" * 30)
            print(f"==> Eval at: {idx + 1}/{len(automerging_parameter_combinations)} ")
            print(f"Memory usage: { psutil.virtual_memory().percent}")

            app_id = f"App-Automerging-chunkSizes={check_sizes}-simTopK={similarity_top_k}-topN={rerank_top_n}"

            query_engine = AutoMergingRetrievalPipeline.from_default_query_engine(
                chunk_sizes=check_sizes,
                similarity_top_k=similarity_top_k,
                rerank_top_n=rerank_top_n,
                documents=cls.documents,
                persist_dir=f"./data/indexes/{app_id}",
            )

            tru_recorder = TruLlama(
                query_engine, app_id=app_id, feedbacks=cls.get_feedbacks()
            )

            for eval_que in tqdm(cls.eval_questions):
                with tru_recorder as recorder:
                    query_engine.query(eval_que)

            del query_engine._node_postprocessors
            print("=" * 30)

        print("------------------ Done")
