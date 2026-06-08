const MAIN_RS: &str = include_str!("../src/main.rs");

#[test]
fn live_copy_trade_hot_path_does_not_recompute_or_backfill_scores() {
    let body = function_body(MAIN_RS, "poll_live_whales_once")
        .expect("poll_live_whales_once should exist");

    assert!(
        body.contains("run_copy_trade_signal_engine"),
        "guard should target the live copy-trade signal hot path"
    );

    for forbidden_call in [
        "backfill_trade_positions_from_copy_signals",
        "enqueue_and_spawn_with_venue",
        "ensure_live_wallet_performance",
        "fetch_closed_positions_for_wallet",
        "recompute_expectancy_flow_cells",
        "recompute_mrs_scores_from_existing",
        "recompute_wallet_segment_scores_from_existing",
        "recompute_wallet_segment_v2_scores_from_existing",
        "score_closed_position_performance",
        "score_mrs",
    ] {
        assert!(
            !body.contains(forbidden_call),
            "live copy-trade hot path must not call `{forbidden_call}`"
        );
    }
}

fn function_body<'a>(source: &'a str, name: &str) -> Option<&'a str> {
    let name_index = source.find(&format!("fn {name}("))?;
    let body_start = source[name_index..].find('{')? + name_index;
    let mut depth = 0usize;

    for (offset, byte) in source[body_start..].bytes().enumerate() {
        match byte {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return source.get(body_start..=body_start + offset);
                }
            }
            _ => {}
        }
    }

    None
}
