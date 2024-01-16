from trulens_eval import Feedback, TruLlama, OpenAI, Tru
from trulens_eval.feedback import Groundedness
import numpy as np
from typing import List
from trulens_utils.utils.constant import Constant as C

from langchain_llama.prompt.variables import (
    CONVERSATION_PROMPT_VARIABLES,
    CONVERSATION_PROMPT_PARTIAL_VARIABLES,
    CONVERSATION_TOOLS_PROMPT_PARTIAL_VARIABLES,
    CONVERSATION_TOOLS_PROMPT_VARIABLES,
    AI_PREFIX,
)
import warnings

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

        print("==============", type(cls.rag_chain))
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


class LlamaindexTrulensEval:
    def __init__(
        cls,
        query_engine,
        app_id: str,
        database_file: str,
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


if __name__ == "__main__":
    from langchain_llama.agent.sales_gpt import SalesGPT
    from trulens_utils.utils.evals_questions import EVAL_QUESTION

    chain = SalesGPT.from_llm(
        verbose=True,
        use_tools=True,
    )

    # print(
    #     chain.sales_agent_executor.invoke(
    #         dict(
    #             input="",
    #             conversation_stage="1",
    #             conversation_history=f"user: {EVAL_QUESTION[0]}",
    #             **CONVERSATION_TOOLS_PROMPT_VARIABLES,
    #             conversation_stages="\n",
    #         )
    #     )
    # )

    trueval = LangchainTrulensEval(
        rag_chain=chain.sales_agent_executor,
        database_file="langchain_turlens",
        app_id="langchain",
        fresh=True,
    )

    # trueval.run_evals(eval_questions=EVAL_QUESTION)
