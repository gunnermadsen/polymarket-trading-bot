"""Export frozen champions into the existing immutable runtime package contract.

No fit, calibration, threshold selection, or qualification changes occur here.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import polars as pl
from .payoff_runtime_export import _histogram, _base_model
from .runtime_export import canonical_json_bytes, write_immutable_directory
from .core_extract import file_sha256

CHAMPIONS = (
    "extended_specialist_official", "bridge_aware_specialist", "official_vwap_admission",
    "official_temporal_consensus", "official_high_precision_loss_veto",
)
DERIVED = ("probability", "confidence", "seconds_elapsed", "share_cost", "expected_edge", "price_bucket_index")
SCHEMA = "btc-5m-payoff-aware-frozen-early-features-v1"

def contract(names: list[str]) -> dict:
    def source(slot, product, semantics, required, history, age):
        return dict(slot=slot, product=product, semantics=semantics, required=required, lookback_seconds=history, maximum_age_ms=age)
    result = dict(version="capitonic-unified-model-runtime-v1", adapter="frozen_early_entry", adapter_version=1,
        probability_semantics="probability_up", feature_clock="closed_binance_second_as_of",
        missing_policy="native_missing_branch", qualified_trade_size=5.0,
        inputs=[source("btc_seconds", "binance_spot_btcusdt_one_second_ohlcv", "binance_closed_seconds_prewindow_open_v1", True, 301, 5000),
            source("execution_book", "polymarket_btc_five_minute_orderbooks", "causal_vwap_five_shares_v1", True, 2, 2000),
            source("oracle", "polygon_chainlink_btcusd_oracle", "causal_oracle_rounds_v1", False, 600, 600000),
            source("candles", "chainlink_btcusd_one_minute_candles", "chainlink_ohlc_close_available_120s_v1", False, 3660, 120000)])

    if any(n.startswith("spot_l2_") for n in names):
        result["inputs"].append(source("spot_l2", "binance_spot_btcusdt_l2_one_second_features", "frozen_spot_l2_point_in_time_v1", False, 60, 5000))
    if any(n.startswith("kraken_l2_") for n in names):
        result["inputs"].append(source("kraken_l2", "kraken_btcusd_l2_updates", "frozen_kraken_l2_point_in_time_v1", False, 60, 5000))
    return result

def export(
    root: Path,
    output: Path,
    panel: Path,
    policy_overrides: dict[str, dict] | None = None,
    selected_champions: set[str] | None = None,
) -> list[dict]:
    policy_overrides = policy_overrides or {}
    collection_path=root/"training-results/btc-5m-frozen-champion-collection-20260902/manifest.json"
    collection=json.loads(collection_path.read_text())
    artifacts=[]
    for item in collection["artifacts"][1:]:
        path=root/item["artifact_path"]
        if file_sha256(path)!=item["artifact_sha256"]: raise ValueError("Frozen source artifact checksum mismatch")
        artifacts.append(joblib.load(path))
    extended, admission, robustness=artifacts
    # Read a bounded projected slice from offline Parquet, never the trading database.
    source=pl.scan_parquet(panel)
    source_names=source.collect_schema().names()
    feature_names=list(extended["models"]["extended_specialist_official"].features)
    selected=list(dict.fromkeys(feature_names+["up_ask_vwap_5","down_ask_vwap_5","fee_rate","pm_up_book_age_seconds","pm_down_book_age_seconds","pm_vwap5_overround","seconds_elapsed"]))
    sample=(source.filter(pl.col("seconds_elapsed").is_between(60,85) & pl.col("up_ask_vwap_5").is_not_null() & pl.col("down_ask_vwap_5").is_not_null()).select([n for n in selected if n in source_names]).head(256).collect())
    if sample.height<16: raise ValueError("Insufficient offline reference rows")
    records=[]
    for champion in CHAMPIONS:
        if selected_champions is not None and champion not in selected_champions:
            continue
        directional=extended["models"]["bridge_aware_specialist" if champion=="bridge_aware_specialist" else "extended_specialist_official"]
        learned=admission["admission_models"].get(champion) or robustness["admission_models"].get(champion)
        temporal=robustness["temporal_models"] if champion=="official_temporal_consensus" else []
        owner=extended if champion in CHAMPIONS[:2] else admission if champion==CHAMPIONS[2] else robustness
        frozen=owner.get("programmatic_policies",owner.get("policies"))[champion]["60_89"]
        if frozen.get("abstain",False):raise ValueError("Cannot export an abstaining policy as trading-enabled")
        policy={key:frozen.get(key) for key in ["minimum_confidence","minimum_edge","maximum_share_cost","minimum_admission_probability","minimum_predicted_stress_edge","maximum_predicted_loss"]}
        policy.update(maximum_temporal_std=frozen.get("maximum_probability_std"),minimum_temporal_agreement=frozen.get("minimum_direction_agreement"),execution_reserve_per_share=0.005)
        override = policy_overrides.get(champion, {})
        unknown = set(override) - set(policy)
        if unknown:
            raise ValueError(f"Unknown policy override fields for {champion}: {sorted(unknown)}")
        policy.update(override)
        names=list(directional.features)
        for name in ["up_ask_vwap_5","down_ask_vwap_5","fee_rate","pm_up_book_age_seconds","pm_down_book_age_seconds","pm_vwap5_overround"] + (list(learned.features) if learned else []):
            if name not in names and name not in DERIVED:names.append(name)
        derived=names+[n for n in DERIVED if n not in names]
        definition=dict(contract=contract(names),outcome=_histogram(directional.student,directional.features,names,"regression"),
            temporal=[_histogram(m.student,m.features,names,"regression") for m in temporal],admission=None,policy=policy)
        if learned:
            definition["admission"]={k:_histogram(getattr(learned,k),learned.features,derived,"probability" if k=="profitable" else "regression") for k in ["profitable","stress_edge","loss_severity"]}
        index=1 if owner is extended else 2 if owner is admission else 3
        source_sha=collection["artifacts"][index]["artifact_sha256"]
        key=f"btc-5m-{champion.replace('_','-')}-umr-20260902"
        if override:
            confidence = policy.get("minimum_confidence")
            if confidence is None:
                raise ValueError(f"Policy override for {champion} does not define confidence")
            key += f"-confidence-{round(float(confidence) * 100):03d}"
        payload=_base_model(key,SCHEMA,names,source_sha,dict(kind="unified",definition=definition,prediction_policy={
            "type":"first_confidence_crossing","minimum_seconds_after_open":60,"maximum_seconds_after_open":89,"cadence_seconds":5,
            "early_end_second":None,"early_cadence_seconds":None,"late_start_second":None}))
        payload["provenance"].update(candidate=champion,source_collection_sha256=file_sha256(collection_path),
            directional_artifact_sha256=collection["artifacts"][1]["artifact_sha256"],source_training_run=owner["run_id"],
            producing_commit=owner["producing_commit"],frozen_policy=frozen,policy_override=override,
            policy_override_basis="sealed_60_89_confidence_sweep",export_is_training=False)
        rows=[]
        for i,row in enumerate(sample.to_dicts()):
            if i%4:continue
            values={name:row.get(name,float("nan")) for name in names}
            for name in names:
                if name.startswith("has_"):values[name]=0.0
                if name.startswith("probability_change_"):values[name]=0.0
            # Explicit native-missing cases supplement real finite historical feature rows.
            if i%16==0:
                for name in directional.features:
                    if name.startswith(("oracle_","binance_oracle_","chainlink_candle_")):values[name]=float("nan")
            if not all(values.get(n) is not None and np.isfinite(values[n]) for n in ["up_ask_vwap_5","down_ask_vwap_5","fee_rate"]):continue
            x=np.array([[values[n] if values[n] is not None else np.nan for n in directional.features]],dtype=float)
            p=float(np.clip(directional.student.predict(x)[0],1e-6,1-1e-6));confidence=max(p,1-p)
            cost=float(values["up_ask_vwap_5" if p>=.5 else "down_ask_vwap_5"])
            edge=confidence-cost-float(values["fee_rate"])*cost*(1-cost)-.005
            seconds=int(row["seconds_elapsed"])
            values.update(probability=p,confidence=confidence,seconds_elapsed=seconds,share_cost=cost,expected_edge=edge,price_bucket_index=sum(cost>v for v in [.65,.8,.95]))
            accepted=cost<=policy["maximum_share_cost"]
            if policy["minimum_confidence"] is not None:accepted &= confidence>=policy["minimum_confidence"]
            if policy["minimum_edge"] is not None:accepted &= edge>=policy["minimum_edge"]
            outputs={}
            if learned:
                a=np.array([[values.get(n,np.nan) for n in learned.features]],dtype=float)
                outputs=dict(admission_probability=float(learned.profitable.predict_proba(a)[0,1]),predicted_stress_edge=float(learned.stress_edge.predict(a)[0]),predicted_loss=max(0.0,float(learned.loss_severity.predict(a)[0])))
                accepted &= outputs["admission_probability"]>=policy["minimum_admission_probability"] and outputs["predicted_stress_edge"]>=policy["minimum_predicted_stress_edge"] and outputs["predicted_loss"]<=policy["maximum_predicted_loss"]
            if temporal:
                ps=np.array([p]+[float(np.clip(m.student.predict(x)[0],1e-6,1-1e-6)) for m in temporal]);std=float(ps.std());agreement=float(((ps>=.5)==(p>=.5)).mean())
                outputs.update(temporal_std=std,temporal_agreement=agreement)
                accepted &= std<=policy["maximum_temporal_std"] and agreement>=policy["minimum_temporal_agreement"]
            rows.append(dict(id=f"frozen-{i}",source={"admission_expected":outputs,"accepted":bool(accepted)},seconds_elapsed=seconds,feature_values=[None if values.get(n) is None or not np.isfinite(values[n]) else float(values[n]) for n in names],
                expected=dict(probability_up=p,confidence=confidence,raw_logit=float(np.log(p/(1-p))),action=("up" if p>=.5 else "down") if accepted else "no_trade")))
        if not rows:raise ValueError("No executable reference fixtures")
        model_bytes=canonical_json_bytes(payload)
        golden_bytes=canonical_json_bytes(dict(schema_version="capitonic-btc-payoff-aware-golden-vectors-v1",model_key=key,feature_schema_sha256=payload["features"]["schema_sha256"],vectors=rows))
        manifest=dict(schema_version="capitonic-btc-directional-runtime-manifest-v1",model_key=key,model_file="model.json",model_sha256=hashlib.sha256(model_bytes).hexdigest(),
            golden_vectors_file="golden-vectors.json",golden_vectors_sha256=hashlib.sha256(golden_bytes).hexdigest(),feature_schema_version=SCHEMA,feature_schema_sha256=payload["features"]["schema_sha256"],
            source_freeze_manifest_sha256=file_sha256(collection_path),source_training_model_sha256=source_sha,deployment_scope="paper_only",production_qualified=False,live_capital_allowed=False)
        directory=output/key
        write_immutable_directory(directory,{"model.json":model_bytes,"manifest.json":canonical_json_bytes(manifest),"golden-vectors.json":golden_bytes})
        records.append(dict(candidate=champion,**manifest,reference_cases=len(rows),contract=definition["contract"],policy=policy))
    return records

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root",type=Path,required=True);parser.add_argument("--output",type=Path,required=True);parser.add_argument("--panel",type=Path,required=True)
    parser.add_argument("--policy-overrides", type=Path)
    parser.add_argument("--champion", action="append", choices=CHAMPIONS)
    args=parser.parse_args()
    overrides = json.loads(args.policy_overrides.read_text()) if args.policy_overrides else None
    print(json.dumps(export(args.source_root,args.output,args.panel,overrides,set(args.champion) if args.champion else None),indent=2))
if __name__=="__main__":main()
