import unittest

from btc_updown_ml.artifact import LinearLogitArtifact, schema_canary_v0
from btc_updown_ml.evaluator import evaluate_probability_artifact, score_probability
from btc_updown_ml.trainer import fit_logistic_residual
from helpers import row


class ArtifactTests(unittest.TestCase):
    def test_schema_canary_reproduces_prior(self) -> None:
        artifact = schema_canary_v0(
            task="settlement_probability_residual",
            feature_schema_version="features-v1",
            feature_names=["distance", "seconds"],
        )
        self.assertEqual(
            artifact.feature_schema_sha256,
            "1502f8f90e7104b04834bbc01ab9675b16a70465e2c01eaadf0468fbd720f258",
        )
        self.assertEqual(
            artifact.artifact_sha256,
            "82c212a8bb1ce1705b6b643ecdffc0f5b13143549b71a88184498d08190f5aef",
        )

        single_feature_artifact = schema_canary_v0(
            task="settlement_probability_residual",
            feature_schema_version="features-v1",
            feature_names=["distance"],
        )
        example = row(1)
        self.assertAlmostEqual(
            score_probability(single_feature_artifact, example), example.prior_probability
        )

    def test_artifact_json_round_trip_validates_hash(self) -> None:
        artifact = schema_canary_v0(
            task="fok_fill_probability",
            feature_schema_version="features-v1",
            feature_names=["distance"],
        )
        decoded = LinearLogitArtifact.from_mapping(__import__("json").loads(artifact.to_json()))
        self.assertEqual(decoded.artifact_sha256, artifact.artifact_sha256)

    def test_trainer_emits_candidate_not_fake_canary(self) -> None:
        rows = [row(index, label=1 if index >= 4 else 0) for index in range(8)]
        cutoff = max(row.label_available_at_ms for row in rows)
        artifact = fit_logistic_residual(
            rows,
            model_version="unit-test-candidate",
            training_cutoff_ms=cutoff,
            iterations=25,
        )
        self.assertEqual(artifact.model_version, "unit-test-candidate")
        metrics = evaluate_probability_artifact(artifact, rows)
        self.assertEqual(metrics.market_count, 8)

    def test_residual_trainer_rejects_fok_labels(self) -> None:
        example = row(1)
        invalid = type(example)(**{**example.__dict__, "task": "fok_fill_probability"})
        with self.assertRaisesRegex(ValueError, "only accepts"):
            fit_logistic_residual(
                [invalid],
                model_version="wrong-task",
                training_cutoff_ms=invalid.label_available_at_ms,
            )


if __name__ == "__main__":
    unittest.main()
