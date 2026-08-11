import hashlib
import unittest

from scripts.analyze_retrieval_case import build_case_report
from scripts.analyze_index_language_overlap import build_overlap_report
from scripts.classify_dimensions import build_prompt
from scripts.create_evaluation_split import build_split_manifest, query_ids_for_split
from scripts.evaluate_baseline import build_report
from scripts.evaluate_baseline import MAIN_PATH, load_json
from scripts.evaluate_dimension_ablation import build_dimension_ablation
from scripts.evaluate_oracle import build_oracle_report
from scripts.evaluate_reranker import build_rerank_report
from scripts.evaluate_semantic_retrieval import (
    evaluate_rankings,
    load_content_documents,
    positive_ranked_ids_from_bm25,
    ranked_ids_from_bm25,
    reciprocal_rank_fusion,
    validate_run_scope,
)
from scripts.validate_evaluation import validate_evaluation


class EvaluationTests(unittest.TestCase):
    def test_evaluation_datasets_are_valid(self):
        errors, stats = validate_evaluation()
        self.assertEqual(errors, [])
        self.assertEqual(stats["main"], 40)
        self.assertEqual(stats["abstain"], 10)
        self.assertEqual(stats["context_requirements"]["thread_context"], 10)
        self.assertEqual(stats["single_turn_included"], 30)

        queries = load_json(MAIN_PATH)
        flagged = {
            query["query_id"]
            for query in queries
            if query.get("excluded_from_single_turn", False)
        }
        thread_context = {
            query["query_id"]
            for query in queries
            if query["context_requirement"] == "thread_context"
        }
        self.assertEqual(flagged, thread_context)
        self.assertEqual(
            sum(
                query["question_type"] == "compound" and query["query_id"] not in flagged
                for query in queries
            ),
            6,
        )

    def test_baseline_report_has_expected_denominators(self):
        report = build_report(k=3, threshold=1.0)
        self.assertEqual(report["configuration"]["documents"], 31)
        self.assertEqual(report["configuration"]["main_queries"], 40)
        self.assertEqual(report["configuration"]["queries_with_gold"], 40)
        self.assertEqual(report["configuration"]["abstain_queries"], 10)
        self.assertEqual(len(report["modes"]), 2)

        aggregate = report["modes"][0]["retrieval"]["aggregate"]
        self.assertEqual(aggregate["hit_at_3"]["n"], 40)
        self.assertEqual(aggregate["recall_at_3_macro"]["n"], 40)
        self.assertEqual(aggregate["recall_at_5_macro"]["n"], 40)
        self.assertEqual(aggregate["recall_at_gold_macro"]["n"], 40)
        self.assertEqual(
            aggregate["gold_size_distribution"],
            {"1": 14, "2": 13, "3": 4, "4": 7, "5": 1, "6": 1},
        )
        self.assertEqual(aggregate["recall_at_3_ceiling_macro"]["value"], 0.93375)

        columns = report["modes"][0]["retrieval"]["by_question_column"]
        self.assertEqual(sum(column["eligible_queries"] for column in columns.values()), 40)
        self.assertEqual(
            {name: column["eligible_queries"] for name, column in columns.items()},
            {
                "A_direct": 11,
                "B_alias_or_coreference": 6,
                "C_symptom": 6,
                "D_compound": 9,
                "E_judgment_or_premise": 8,
            },
        )
        self.assertEqual(
            {name: column["total_queries"] for name, column in columns.items()},
            {
                "A_direct": 11,
                "B_alias_or_coreference": 6,
                "C_symptom": 6,
                "D_compound": 9,
                "E_judgment_or_premise": 8,
            },
        )

    def test_structural_abstention_and_answer_shape_are_reported(self):
        report = build_report(k=3, threshold=1.0)
        gate = report["structural_abstention_gate"]
        self.assertEqual(gate["abstain_set_accuracy"]["correct_abstentions"], 10)
        self.assertEqual(gate["abstain_set_accuracy"]["n"], 10)
        self.assertEqual(gate["covered_query_acceptance"]["accepted"], 24)
        self.assertEqual(gate["covered_query_acceptance"]["n"], 30)
        self.assertEqual(gate["context_free_covered_query_acceptance"]["value"], 1.0)
        repotting = next(
            detail for detail in gate["abstain_details"] if detail["query_id"] == "q:abstain:005"
        )
        self.assertEqual(repotting["missing_dimensions"], ["repotting"])
        self.assertEqual(repotting["explanation"], "我还没有龟背竹的换盆资料。")

        raw_shape = report["modes"][0]["answer_shape_analysis"]
        self.assertEqual(raw_shape["shape_gap_queries"], 1)
        self.assertEqual(raw_shape["retrieval_complete_at_3"]["queries_with_all_gold_retrieved"], 0)
        self.assertEqual(raw_shape["retrieval_complete_at_3"]["n"], 1)

        entity_shape = report["modes"][1]["answer_shape_analysis"]
        self.assertEqual(entity_shape["shape_gap_queries"], 1)
        self.assertEqual(entity_shape["retrieval_complete_at_3"]["queries_with_all_gold_retrieved"], 1)
        self.assertEqual(entity_shape["retrieval_complete_at_3"]["n"], 1)

    def test_dimension_oracle_uses_existing_manual_labels(self):
        report = build_oracle_report(k=3)
        self.assertTrue(
            report["label_integrity"]["gold_dimensions_are_covered_by_expected_dimensions"]
        )
        self.assertEqual(report["label_integrity"]["violations"], [])
        scenarios = {scenario["name"]: scenario for scenario in report["scenarios"]}

        dimension_oracle = scenarios["oracle_dimensions_current_entity"]["retrieval"]
        self.assertEqual(
            dimension_oracle["aggregate"]["recall_at_gold_macro"]["value"], 0.722083
        )
        compound = dimension_oracle["by_question_column"]["D_compound"]
        self.assertEqual(compound["hit_at_3"]["hits"], 8)
        self.assertEqual(compound["hit_at_3"]["n"], 9)
        self.assertEqual(compound["recall_at_gold_macro"]["value"], 0.546296)

        entity_oracle = scenarios["oracle_entity_only"]["retrieval"]
        self.assertEqual(entity_oracle["aggregate"]["recall_at_gold_macro"]["value"], 0.60875)
        self.assertEqual(
            entity_oracle["by_question_column"]["D_compound"]["recall_at_gold_macro"]["value"],
            0.648148,
        )

        full_oracle = scenarios["oracle_dimensions_and_entity"]["retrieval"]
        self.assertEqual(full_oracle["aggregate"]["recall_at_gold_macro"]["value"], 0.78875)

    def test_few_shot_prompt_does_not_include_evaluation_labels(self):
        queries = load_json(MAIN_PATH)
        prompt = build_prompt(queries)
        self.assertNotIn("expected_dimensions", prompt)
        self.assertNotIn("gold_entry_ids", prompt)
        self.assertIn(queries[0]["raw_text"], prompt)
        conservative = load_json(
            MAIN_PATH.parents[1] / "results" / "dimension_predictions.json"
        )
        self.assertEqual(
            hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            conservative["prompt_sha256"],
        )

        recall_prompt = build_prompt(queries, prompt_style="recall_first")
        self.assertNotIn("expected_dimensions", recall_prompt)
        self.assertNotIn("gold_entry_ids", recall_prompt)
        self.assertIn("Optimize for recall", recall_prompt)
        recall = load_json(
            MAIN_PATH.parents[1] / "results" / "dimension_predictions_recall.json"
        )
        self.assertEqual(
            hashlib.sha256(recall_prompt.encode("utf-8")).hexdigest(),
            recall["prompt_sha256"],
        )

    def test_recorded_dimension_predictions_and_ablation(self):
        payload = load_json(MAIN_PATH.parents[1] / "results" / "dimension_predictions.json")
        report = build_dimension_ablation(payload, k=3)
        micro = report["dimension_classification"]["micro"]
        self.assertEqual(micro, {
            "precision": 1.0,
            "recall": 0.875,
            "f1": 0.933333,
            "tp": 56,
            "fp": 0,
            "fn": 8,
        })
        self.assertEqual(report["dimension_classification"]["exact_match"]["exact"], 32)
        stages = {stage["name"]: stage for stage in report["retrieval_ablation"]}
        self.assertEqual(
            stages["bm25_entity_predicted_dimensions"]["retrieval"]["aggregate"]
            ["recall_at_gold_macro"]["value"],
            0.7075,
        )
        compound = stages["bm25_entity_predicted_dimensions"]["retrieval"]
        compound = compound["by_question_column"]["D_compound"]
        self.assertEqual(compound["hit_at_3"]["hits"], 8)
        self.assertEqual(compound["hit_at_3"]["n"], 9)
        self.assertEqual(compound["recall_at_5_macro"]["value"], 0.638889)

    def test_recall_first_predictions_pass_the_predeclared_stop_condition(self):
        payload = load_json(
            MAIN_PATH.parents[1] / "results" / "dimension_predictions_recall.json"
        )
        report = build_dimension_ablation(payload, k=3)
        self.assertEqual(
            report["dimension_classification"]["micro"],
            {
                "precision": 1.0,
                "recall": 0.921875,
                "f1": 0.95935,
                "tp": 59,
                "fp": 0,
                "fn": 5,
            },
        )
        self.assertEqual(report["dimension_classification"]["exact_match"]["exact"], 35)
        predicted = next(
            stage
            for stage in report["retrieval_ablation"]
            if stage["name"] == "bm25_entity_predicted_dimensions"
        )["retrieval"]
        self.assertGreater(predicted["aggregate"]["recall_at_5_macro"]["value"], 0.87)
        self.assertEqual(predicted["aggregate"]["recall_at_5_macro"]["value"], 0.87625)
        self.assertEqual(predicted["aggregate"]["recall_at_gold_macro"]["value"], 0.709583)

    def test_persistent_compound_failure_has_reproducible_case_evidence(self):
        payload = load_json(
            MAIN_PATH.parents[1] / "results" / "dimension_predictions_recall.json"
        )
        report = build_case_report("q:main:031", payload, top_n=12)
        self.assertEqual(report["query"]["answer_shape"], "diagnosis_list")
        self.assertEqual(report["query"]["context_requirement"], "thread_context")
        self.assertTrue(report["query"]["excluded_from_single_turn"])
        self.assertIsNone(report["query"]["detected_plant_id"])
        scenarios = {scenario["name"]: scenario for scenario in report["rankings"]}
        self.assertEqual(scenarios["bm25_raw"]["first_gold_rank"], 7)
        self.assertEqual(
            scenarios["bm25_entity_oracle_dimensions"]["first_gold_rank"], 6
        )
        self.assertEqual(scenarios["bm25_oracle_entity"]["first_gold_rank"], 3)
        overlaps = {
            item["entry_id"]: item for item in report["gold_lexical_overlap"]
        }
        self.assertEqual(overlaps["kb:pothos_lighting_001"]["overlap_count"], 0)
        self.assertEqual(
            overlaps["kb:pothos_symptom_overwatering_001"]["overlap_tokens"],
            ["leaf"],
        )

    def test_rerank_report_preserves_full_and_uniform_single_turn_populations(self):
        payload = load_json(
            MAIN_PATH.parents[1] / "results" / "dimension_predictions_recall.json"
        )

        def tied_scorer(pairs):
            return [0.0] * len(pairs)

        report = build_rerank_report(
            payload,
            tied_scorer,
            {"model_id": "test/tied-scorer"},
            k=3,
        )
        self.assertEqual(report["configuration"]["full_queries"], 40)
        self.assertEqual(report["configuration"]["single_turn_queries"], 30)
        self.assertEqual(report["configuration"]["excluded_thread_context_queries"], 10)
        self.assertEqual(report["configuration"]["full_compound_queries"], 9)
        self.assertEqual(report["configuration"]["single_turn_compound_queries"], 6)

        for stage in report["retrieval_ablation"]:
            self.assertEqual(stage["retrieval"]["aggregate"]["eligible_queries"], 40)
            self.assertEqual(
                stage["retrieval_single_turn"]["aggregate"]["eligible_queries"], 30
            )
            self.assertEqual(
                stage["retrieval"]["by_question_column"]["D_compound"]
                ["hit_at_3"]["n"],
                9,
            )
            self.assertEqual(
                stage["retrieval_single_turn"]["by_question_column"]["D_compound"]
                ["hit_at_3"]["n"],
                6,
            )

        stages = {stage["name"]: stage for stage in report["retrieval_ablation"]}
        lexical = stages["bm25_entity_predicted_dimensions"]["retrieval"]["aggregate"]
        reranked = stages["cross_encoder_entity_predicted_dimensions"]["retrieval"]["aggregate"]
        self.assertEqual(
            lexical["recall_at_gold_macro"], reranked["recall_at_gold_macro"]
        )

    def test_recorded_cross_encoder_result_is_reproducible_without_model_download(self):
        report = load_json(MAIN_PATH.parents[1] / "results" / "rerank_ablation.json")
        self.assertEqual(
            report["model"]["model_id"], "cross-encoder/ms-marco-MiniLM-L6-v2"
        )
        self.assertEqual(
            report["model"]["model_revision"],
            "233902d25c440f23af6f7d6e94d2946bac0bee0a",
        )
        stages = {stage["name"]: stage for stage in report["retrieval_ablation"]}
        predicted = stages["cross_encoder_entity_predicted_dimensions"]
        self.assertEqual(
            predicted["retrieval"]["aggregate"]["recall_at_gold_macro"]["value"],
            0.728333,
        )
        self.assertEqual(
            predicted["retrieval_single_turn"]["aggregate"]
            ["recall_at_gold_macro"]["value"],
            0.798889,
        )
        self.assertEqual(
            predicted["retrieval"]["by_question_column"]["D_compound"]
            ["hit_at_3"],
            {"value": 0.888889, "hits": 8, "n": 9},
        )
        self.assertEqual(
            predicted["retrieval_single_turn"]["by_question_column"]["D_compound"]
            ["hit_at_3"],
            {"value": 1.0, "hits": 6, "n": 6},
        )

    def test_index_language_overlap_is_metadata_dominated(self):
        report = build_overlap_report()
        self.assertEqual(report["population"]["query_gold_pairs"], 91)
        overlap = report["pair_overlap"]
        self.assertEqual(overlap["metadata_any"]["count"], 88)
        self.assertEqual(overlap["content_any"]["count"], 4)
        self.assertEqual(overlap["metadata_only"]["count"], 84)
        self.assertEqual(overlap["content_only"]["count"], 0)
        self.assertEqual(overlap["both"]["count"], 4)
        self.assertEqual(overlap["neither"]["count"], 3)
        self.assertEqual(
            report["query_coverage"]["with_any_gold_metadata_overlap"]["count"], 40
        )

    def test_dev_test_split_is_deterministic_complete_and_label_blind(self):
        queries = load_json(MAIN_PATH)
        manifest = build_split_manifest(queries, dev_size=20)
        repeated = build_split_manifest(list(reversed(queries)), dev_size=20)
        self.assertEqual(manifest["assignments_by_hash"], repeated["assignments_by_hash"])
        self.assertFalse(manifest["algorithm"]["uses_query_text"])
        self.assertFalse(manifest["algorithm"]["uses_labels_or_metrics"])

        dev = query_ids_for_split(manifest, "dev")
        test = query_ids_for_split(manifest, "test")
        all_ids = {query["query_id"] for query in queries}
        self.assertEqual(len(dev), 20)
        self.assertEqual(len(test), 20)
        self.assertFalse(dev & test)
        self.assertEqual(dev | test, all_ids)

        recorded = load_json(
            MAIN_PATH.parents[1] / "splits" / "main_sha256_20_20_v1.json"
        )
        self.assertEqual(recorded, manifest)

    def test_content_only_bm25_floor_and_semantic_test_gate_are_explicit(self):
        queries = load_json(MAIN_PATH)
        manifest = load_json(
            MAIN_PATH.parents[1] / "splits" / "main_sha256_20_20_v1.json"
        )
        dev_ids = query_ids_for_split(manifest, "dev")
        dev = [query for query in queries if query["query_id"] in dev_ids]
        documents = load_content_documents()

        forced = evaluate_rankings(dev, ranked_ids_from_bm25(dev, documents), k=3)
        signal_only_rankings = positive_ranked_ids_from_bm25(dev, documents)
        signal_only = evaluate_rankings(dev, signal_only_rankings, k=3)
        self.assertEqual(
            forced["retrieval"]["aggregate"]["recall_at_5_macro"]["value"],
            0.225,
        )
        self.assertEqual(
            signal_only["retrieval"]["aggregate"]["recall_at_5_macro"]["value"],
            0.05,
        )
        self.assertEqual(
            signal_only["retrieval"]["query_gold_pair_recall_at_5"],
            {"value": 0.022222, "hits": 1, "n": 45},
        )
        self.assertEqual(sum(bool(ranking) for ranking in signal_only_rankings.values()), 6)

        self.assertEqual(
            reciprocal_rank_fusion([["a", "b"], ["b", "c"]], rrf_k=60),
            ["b", "a", "c"],
        )
        self.assertEqual(
            reciprocal_rank_fusion(
                [["a", "b"], ["b", "c"]], rrf_k=60, weights=[1.0, 2.0]
            ),
            ["b", "c", "a"],
        )
        with self.assertRaises(ValueError):
            reciprocal_rank_fusion([["a"], ["b"]], weights=[1.0])
        with self.assertRaises(ValueError):
            reciprocal_rank_fusion([["a"]], weights=[0.0])
        validate_run_scope("dev", False, None)
        with self.assertRaises(ValueError):
            validate_run_scope("test", False, ["BAAI/bge-m3"])
        with self.assertRaises(ValueError):
            validate_run_scope(
                "test", True, ["BAAI/bge-m3", "intfloat/multilingual-e5-large"]
            )
        validate_run_scope("test", True, ["BAAI/bge-m3"])

    def test_recorded_semantic_dev_result_never_touches_test(self):
        report = load_json(
            MAIN_PATH.parents[1] / "results" / "semantic_retrieval_dev.json"
        )
        self.assertEqual(report["population"]["split"], "dev")
        self.assertEqual(report["population"]["queries"], 20)
        self.assertFalse(report["population"]["test_queries_evaluated_or_encoded"])

        manifest = load_json(
            MAIN_PATH.parents[1] / "splits" / "main_sha256_20_20_v1.json"
        )
        dev_ids = query_ids_for_split(manifest, "dev")
        test_ids = query_ids_for_split(manifest, "test")
        result_ids = {
            row["query_id"]
            for row in report["baselines"]["metadata_plus_content_bm25"]
            ["retrieval"]["per_query"]
        }
        self.assertEqual(result_ids, dev_ids)
        self.assertFalse(result_ids & test_ids)

        positive = report["baselines"]["content_only_bm25_positive_score_only"]
        self.assertEqual(
            positive["retrieval"]["aggregate"]["recall_at_5_macro"]["value"],
            0.05,
        )
        self.assertEqual(
            positive["retrieval"]["query_gold_pair_recall_at_5"],
            {"value": 0.022222, "hits": 1, "n": 45},
        )

        candidates = {
            candidate["model"]["model_id"]: candidate
            for candidate in report["embedding_candidates"]
        }
        self.assertEqual(set(candidates), {
            "BAAI/bge-m3",
            "intfloat/multilingual-e5-large",
        })
        self.assertEqual(
            candidates["BAAI/bge-m3"]["model"]["model_revision"],
            "5617a9f61b028005a4858fdac845db406aefb181",
        )
        self.assertEqual(
            candidates["intfloat/multilingual-e5-large"]["model"]["model_revision"],
            "3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3",
        )
        self.assertEqual(
            candidates["BAAI/bge-m3"]["evaluation"]["retrieval"]
            ["aggregate"]["recall_at_5_macro"]["value"],
            0.713333,
        )
        self.assertEqual(
            candidates["intfloat/multilingual-e5-large"]["evaluation"]["retrieval"]
            ["aggregate"]["recall_at_5_macro"]["value"],
            0.696667,
        )
        self.assertEqual(report["selection"]["selected_model_id"], "BAAI/bge-m3")

        hybrid = report["hybrid_selected_embedding_rrf"]["evaluation"]
        self.assertEqual(
            hybrid["retrieval"]["aggregate"]["recall_at_5_macro"]["value"],
            0.750833,
        )
        self.assertEqual(
            hybrid["retrieval_single_turn"]["aggregate"]
            ["recall_at_5_macro"]["value"],
            0.856667,
        )
        self.assertEqual(
            report["channel_complementarity_at_5"],
            {
                "metadata_plus_content_bm25_pairs": 26,
                "content_embedding_pairs": 26,
                "intersection_pairs": 19,
                "bm25_only_pairs": 7,
                "embedding_only_pairs": 7,
                "union_pairs": 33,
                "fixed_rrf_pairs": 26,
                "total_gold_pairs": 45,
            },
        )

    def test_fusion_weight_sweep_obeys_the_pre_registered_stop_condition(self):
        experiment = load_json(
            MAIN_PATH.parents[1] / "experiments" / "fusion_weight_sweep_v1.json"
        )
        report = load_json(
            MAIN_PATH.parents[1] / "results" / "fusion_weight_sweep_dev.json"
        )
        self.assertEqual(
            experiment["only_tunable_parameter"]["embedding_weight_grid"],
            [0.5, 0.75, 1.0, 1.25, 1.5],
        )
        self.assertFalse(report["population"]["test_queries_evaluated_or_encoded"])
        observed = {
            candidate["embedding_weight"]: candidate["evaluation"]["retrieval"]
            ["query_gold_pair_recall_at_5"]["hits"]
            for candidate in report["weight_candidates"]
        }
        self.assertEqual(observed, {0.5: 28, 0.75: 27, 1.0: 26, 1.25: 26, 1.5: 27})
        selection = report["selection_result"]
        self.assertTrue(selection["stop_condition_met"])
        self.assertEqual(selection["decision"], "accept_original_equal_weight_rrf")
        self.assertEqual(selection["final_embedding_weight"], 1.0)

    def test_recorded_semantic_test_result_matches_the_frozen_one_time_run(self):
        report = load_json(
            MAIN_PATH.parents[1] / "results" / "semantic_retrieval_test.json"
        )
        run = load_json(
            MAIN_PATH.parents[1] / "results" / "semantic_retrieval_test_run_v1.json"
        )
        manifest = load_json(
            MAIN_PATH.parents[1] / "splits" / "main_sha256_20_20_v1.json"
        )
        test_ids = query_ids_for_split(manifest, "test")
        dev_ids = query_ids_for_split(manifest, "dev")
        result_ids = {
            row["query_id"]
            for row in report["baselines"]["metadata_plus_content_bm25"]
            ["retrieval"]["per_query"]
        }
        self.assertEqual(report["population"]["split"], "test")
        self.assertTrue(report["population"]["test_queries_evaluated_or_encoded"])
        self.assertEqual(result_ids, test_ids)
        self.assertFalse(result_ids & dev_ids)
        self.assertEqual(run["execution_count"], 1)
        self.assertFalse(run["test_result_existed_before_run"])
        self.assertEqual(
            report["embedding_candidates"][0]["model"]["model_revision"],
            "5617a9f61b028005a4858fdac845db406aefb181",
        )
        self.assertEqual(report["hybrid_selected_embedding_rrf"]["rrf_k"], 60)
        self.assertEqual(report["hybrid_selected_embedding_rrf"]["weights"], [1.0, 1.0])

        content_bm25 = report["baselines"]["content_only_bm25_positive_score_only"]
        content_bge = report["embedding_candidates"][0]["evaluation"]
        mixed_bm25 = report["baselines"]["metadata_plus_content_bm25"]
        hybrid = report["hybrid_selected_embedding_rrf"]["evaluation"]
        self.assertEqual(
            content_bm25["retrieval"]["aggregate"]["recall_at_5_macro"]["value"],
            0.15,
        )
        self.assertEqual(
            content_bge["retrieval"]["aggregate"]["recall_at_5_macro"]["value"],
            0.654167,
        )
        self.assertEqual(
            mixed_bm25["retrieval"]["aggregate"]["recall_at_5_macro"]["value"],
            0.783333,
        )
        self.assertEqual(
            hybrid["retrieval"]["aggregate"]["recall_at_5_macro"]["value"],
            0.770833,
        )
        self.assertEqual(
            hybrid["retrieval"]["aggregate"]["recall_at_gold_macro"]["value"],
            0.583333,
        )
        self.assertEqual(
            report["channel_complementarity_at_5"],
            {
                "metadata_plus_content_bm25_pairs": 32,
                "content_embedding_pairs": 26,
                "intersection_pairs": 24,
                "bm25_only_pairs": 8,
                "embedding_only_pairs": 2,
                "union_pairs": 34,
                "fixed_rrf_pairs": 32,
                "total_gold_pairs": 46,
            },
        )


if __name__ == "__main__":
    unittest.main()
