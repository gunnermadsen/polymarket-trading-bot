from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .dataset import SnapshotRow


@dataclass(frozen=True)
class WalkForwardFold:
    fold_index: int
    training_cutoff_ms: int
    train: tuple[SnapshotRow, ...]
    calibration: tuple[SnapshotRow, ...]
    test: tuple[SnapshotRow, ...]

    def validate(self) -> None:
        train_markets = {row.market_id for row in self.train}
        calibration_markets = {row.market_id for row in self.calibration}
        test_markets = {row.market_id for row in self.test}
        if train_markets & calibration_markets:
            raise ValueError("train and calibration markets overlap")
        if train_markets & test_markets:
            raise ValueError("train and test markets overlap")
        if calibration_markets & test_markets:
            raise ValueError("calibration and test markets overlap")
        if any(not row.eligible_for_training(self.training_cutoff_ms) for row in self.train):
            raise ValueError("training fold contains a label unavailable at its cutoff")
        if max(row.feature_as_of_ms for row in self.train) >= min(
            row.feature_as_of_ms for row in self.calibration
        ):
            raise ValueError("training features are not chronological")
        if max(row.feature_as_of_ms for row in self.calibration) >= min(
            row.feature_as_of_ms for row in self.test
        ):
            raise ValueError("calibration features are not chronological")


@dataclass(frozen=True)
class _MarketGroup:
    market_id: str
    rows: tuple[SnapshotRow, ...]
    first_feature_ms: int
    last_feature_ms: int
    label_available_at_ms: int


def chronological_grouped_walk_forward(
    rows: Sequence[SnapshotRow],
    *,
    min_train_markets: int,
    calibration_markets: int,
    test_markets: int,
    step_markets: int | None = None,
    purge_ms: int = 0,
) -> list[WalkForwardFold]:
    if min_train_markets < 1 or calibration_markets < 1 or test_markets < 1:
        raise ValueError("train, calibration, and test market counts must be positive")
    if purge_ms < 0:
        raise ValueError("purge_ms must not be negative")
    step = test_markets if step_markets is None else step_markets
    if step < 1:
        raise ValueError("step_markets must be positive")

    groups = _group_by_market(rows)
    folds: list[WalkForwardFold] = []
    boundary = min_train_markets + calibration_markets
    while boundary + test_markets <= len(groups):
        calibration_start = boundary - calibration_markets
        calibration_groups_slice = groups[calibration_start:boundary]
        test_groups_slice = groups[boundary : boundary + test_markets]
        calibration_start_ms = calibration_groups_slice[0].first_feature_ms
        test_start_ms = test_groups_slice[0].first_feature_ms
        training_cutoff_ms = calibration_start_ms - purge_ms

        train_groups = tuple(
            group
            for group in groups[:calibration_start]
            if group.last_feature_ms < training_cutoff_ms
            and group.label_available_at_ms <= training_cutoff_ms
        )
        eligible_calibration = tuple(
            group
            for group in calibration_groups_slice
            if group.last_feature_ms < test_start_ms - purge_ms
            and group.label_available_at_ms <= test_start_ms - purge_ms
        )
        if len(train_groups) >= min_train_markets and len(eligible_calibration) == calibration_markets:
            fold = WalkForwardFold(
                fold_index=len(folds),
                training_cutoff_ms=training_cutoff_ms,
                train=_flatten(train_groups),
                calibration=_flatten(eligible_calibration),
                test=_flatten(tuple(test_groups_slice)),
            )
            fold.validate()
            folds.append(fold)
        boundary += step
    return folds


def _group_by_market(rows: Sequence[SnapshotRow]) -> list[_MarketGroup]:
    if not rows:
        raise ValueError("rows must not be empty")
    grouped: dict[str, list[SnapshotRow]] = {}
    for row in rows:
        row.validate()
        grouped.setdefault(row.market_id, []).append(row)

    groups: list[_MarketGroup] = []
    for market_id, market_rows in grouped.items():
        market_rows.sort(key=lambda row: (row.feature_as_of_ms, row.snapshot_id))
        labels = {row.label for row in market_rows}
        label_times = {row.label_available_at_ms for row in market_rows}
        if len(labels) != 1 or len(label_times) != 1:
            raise ValueError(f"market {market_id!r} has inconsistent labels")
        groups.append(
            _MarketGroup(
                market_id=market_id,
                rows=tuple(market_rows),
                first_feature_ms=market_rows[0].feature_as_of_ms,
                last_feature_ms=market_rows[-1].feature_as_of_ms,
                label_available_at_ms=market_rows[0].label_available_at_ms,
            )
        )
    groups.sort(key=lambda group: (group.first_feature_ms, group.market_id))
    for left, right in zip(groups, groups[1:]):
        if left.last_feature_ms >= right.first_feature_ms:
            raise ValueError(
                "market feature windows overlap; define a market-level ordering policy before splitting"
            )
    return groups


def _flatten(groups: tuple[_MarketGroup, ...]) -> tuple[SnapshotRow, ...]:
    return tuple(row for group in groups for row in group.rows)

