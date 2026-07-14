import unittest

from btc_updown_ml.splits import chronological_grouped_walk_forward
from helpers import row


class WalkForwardTests(unittest.TestCase):
    def test_folds_never_overlap_markets(self) -> None:
        rows = [row(index) for index in range(10)]
        folds = chronological_grouped_walk_forward(
            rows,
            min_train_markets=3,
            calibration_markets=1,
            test_markets=2,
            step_markets=2,
            purge_ms=100,
        )
        self.assertGreater(len(folds), 0)
        for fold in folds:
            train = {item.market_id for item in fold.train}
            calibration = {item.market_id for item in fold.calibration}
            test = {item.market_id for item in fold.test}
            self.assertFalse(train & calibration)
            self.assertFalse(train & test)
            self.assertFalse(calibration & test)
            self.assertTrue(
                all(item.label_available_at_ms <= fold.training_cutoff_ms for item in fold.train)
            )


if __name__ == "__main__":
    unittest.main()

