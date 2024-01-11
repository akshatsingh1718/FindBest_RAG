from trulens_eval import Feedback, TruLlama, OpenAI, Tru
from trulens_eval.feedback import Groundedness
import numpy as np
from typing import List


class LlamaTrulensEval:
    def __init__(
        cls,
        query_engine,
        app_id: str,
        database_file: str = "",
        verbose=False,
        fresh=False,
    ):
        cls.query_engine = query_engine
        cls.app_id = app_id
        cls.verbose = verbose
        cls.fresh = fresh
        cls.database_file = database_file or "default.sqlite"

        if cls.fresh:
            if cls.verbose:
                print(f"Resetting database:")
            Tru(database_file=cls.database_file).reset_database()

        cls.trulens_recorder = cls.get_triad_trulens_recorder()
        

    def get_triad_trulens_recorder(cls):
        openai = OpenAI()

        qa_relevance = Feedback(
            openai.relevance_with_cot_reasons, name="Answer Relevance"
        ).on_input_output()

        qs_relevance = (
            Feedback(openai.relevance_with_cot_reasons, name="Context Relevance")
            .on_input()
            .on(TruLlama.select_source_nodes().node.text)
            .aggregate(np.mean)
        )

        grounded = Groundedness(groundedness_provider=openai)

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

    def run_evals(cls, eval_questions: List[str], start_dashboard=True):
        if start_dashboard:
            Tru(database_file=cls.database_file).run_dashboard()
        for question in eval_questions:
            with cls.trulens_recorder as recording:
                response = cls.query_engine.query(question)
                if cls.verbose:
                    print(response)
