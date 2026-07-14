import hashlib
import unittest

from btc_updown_ml.artifact import canonical_feature_schema
from btc_updown_ml.dataset import feature_vector_sha256
from btc_updown_ml.evaluator import score_probability
from btc_updown_ml.runtime_contract import (
    artifact_from_contract,
    load_runtime_contract_fixture,
    snapshot_from_contract,
)


class RuntimeContractTests(unittest.TestCase):
    def test_runtime_v2_contract_matches_all_pinned_hashes(self) -> None:
        fixture = load_runtime_contract_fixture()
        manifest = fixture["manifest"]
        self.assertEqual(
            hashlib.sha256(manifest["seed"].encode("utf-8")).hexdigest(),
            "d9ba98003651835e4c5591f7818076e13d1b0fc16eab1683fb47f917c58a45c6",
        )
        self.assertEqual(manifest["sha256"], "d9ba98003651835e4c5591f7818076e13d1b0fc16eab1683fb47f917c58a45c6")

        artifacts = {
            specification["task"]: artifact_from_contract(
                specification, manifest["sha256"]
            )
            for specification in fixture["artifacts"]
        }
        self.assertEqual(
            set(artifacts),
            {
                "settlement_probability_residual",
                "fok_fill_probability",
                "fok_post_fill_toxicity_probability",
            },
        )
        for specification in fixture["artifacts"]:
            artifact = artifacts[specification["task"]]
            schema_hash = hashlib.sha256(
                canonical_feature_schema(
                    specification["feature_schema_version"],
                    specification["feature_names"],
                ).encode("utf-8")
            ).hexdigest()
            self.assertEqual(
                schema_hash, specification["expected_feature_schema_sha256"]
            )
            self.assertEqual(
                artifact.feature_schema_sha256,
                specification["expected_feature_schema_sha256"],
            )
            self.assertEqual(
                artifact.artifact_sha256,
                specification["expected_artifact_sha256"],
            )
            self.assertEqual(artifact.dataset_manifest_sha256, manifest["sha256"])

        vector_specification = fixture["vector"]
        snapshot = snapshot_from_contract(vector_specification)
        self.assertEqual(
            feature_vector_sha256(snapshot),
            vector_specification["expected_vector_sha256"],
        )
        self.assertEqual(
            artifacts[snapshot.task].feature_schema_sha256,
            vector_specification["expected_feature_schema_sha256"],
        )
        self.assertAlmostEqual(
            score_probability(artifacts[snapshot.task], snapshot),
            vector_specification["expected_probability"],
        )


if __name__ == "__main__":
    unittest.main()
