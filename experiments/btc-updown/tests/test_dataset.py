import unittest

from btc_updown_ml.dataset import (
    FeatureObservation,
    SnapshotRow,
    build_dataset_manifest,
    feature_vector_sha256,
)
from helpers import row


class DatasetTests(unittest.TestCase):
    def test_future_source_event_is_rejected(self) -> None:
        invalid = SnapshotRow(
            snapshot_id="snapshot",
            market_id="market",
            feature_as_of_ms=100,
            feature_received_at_ms=100,
            label_available_at_ms=200,
            label=1,
            task="settlement_probability_residual",
            label_version="label-v1",
            prior_probability=0.5,
            schema_version="features-v1",
            features=(FeatureObservation("distance", 1.0, 101, 99),),
        )
        with self.assertRaisesRegex(ValueError, "source event"):
            invalid.validate()

    def test_manifest_rejects_label_unavailable_at_cutoff(self) -> None:
        example = row(1)
        with self.assertRaisesRegex(ValueError, "unavailable"):
            build_dataset_manifest([example], training_cutoff_ms=example.label_available_at_ms - 1)

    def test_manifest_hash_is_deterministic(self) -> None:
        rows = [row(1), row(2)]
        cutoff = max(item.label_available_at_ms for item in rows)
        left = build_dataset_manifest(rows, training_cutoff_ms=cutoff)
        right = build_dataset_manifest(list(reversed(rows)), training_cutoff_ms=cutoff)
        self.assertEqual(left.dataset_sha256, right.dataset_sha256)
        self.assertEqual(left.manifest_sha256, right.manifest_sha256)
        self.assertNotEqual(left.dataset_sha256, left.manifest_sha256)

    def test_post_label_features_are_rejected(self) -> None:
        example = row(1)
        invalid = SnapshotRow(
            **{
                **example.__dict__,
                "feature_as_of_ms": example.label_available_at_ms,
                "feature_received_at_ms": example.label_available_at_ms,
            }
        )
        with self.assertRaisesRegex(ValueError, "before the label"):
            invalid.validate()

    def test_feature_vector_hash_is_stable(self) -> None:
        self.assertEqual(feature_vector_sha256(row(1)), feature_vector_sha256(row(1)))


if __name__ == "__main__":
    unittest.main()
