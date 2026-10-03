//! Empirical verification of node locking semantics (Phase 1 of the Desktop Postflop nodelock work).
//!
//! Checks:
//! 1. A locked node reports exactly the locked frequencies (`strategy()` is lock-aware).
//! 2. Lock orientation: navigating to the IP decision node with `play(1)` and locking
//!    fold-heavy / call-heavy produces the expected EV shift for OOP.
//! 3. The exploitability returned by `solve()` (and `compute_exploitability()`) is
//!    *restricted* exploitability: it is the best response with the locked node constrained,
//!    therefore it still converges to ~0 even when locks are present.
//! 4. `compute_current_ev()` is lock-aware (root EV = EV of the locked profile).

use postflop_solver::*;

const FOLD: usize = 0;
const CALL: usize = 1;

fn build_game() -> PostFlopGame {
    let card_config = CardConfig {
        range: ["AsAh,QsQh".parse().unwrap(), "KsKh".parse().unwrap()],
        flop: flop_from_str("2s3h4d").unwrap(),
        turn: card_from_str("6c").unwrap(),
        river: card_from_str("7c").unwrap(),
    };
    let tree_config = TreeConfig {
        initial_state: BoardState::River,
        starting_pot: 20,
        effective_stack: 10,
        river_bet_sizes: [("a", "").try_into().unwrap(), ("a", "").try_into().unwrap()],
        ..Default::default()
    };
    let action_tree = ActionTree::new(tree_config).unwrap();
    PostFlopGame::with_config(card_config, action_tree).unwrap()
}

/// Solves, optionally locking the IP node reached by OOP's all-in.
/// `lock` = (fold frequency, call frequency) for IP's single hand, or None for the baseline.
fn solve_with_ip_lock(lock: Option<(f32, f32)>) -> (f32, [f32; 2], Vec<f32>) {
    let mut game = build_game();
    game.allocate_memory(false);
    if let Some((fold, call)) = lock {
        game.play(1); // OOP all-in -> IP decision node (fold / call)
        game.lock_current_strategy(&[fold, call]); // layout: action * num_hands + hand
        game.back_to_root();
    }
    let exploitability = solve(&mut game, 2000, 1e-4, false);
    game.cache_normalized_weights();
    let ev = compute_current_ev(&game);
    assert!(
        (ev[0] + ev[1]).abs() < 1e-3,
        "EV should be zero-sum, got {ev:?}"
    );

    // reported (lock-aware) strategy at the IP node
    game.play(1);
    let ip_strategy = game.strategy();
    game.back_to_root();
    (exploitability, ev, ip_strategy)
}

#[test]
fn baseline_has_no_lock_and_converges() {
    let (expl, ev, ip_strategy) = solve_with_ip_lock(None);
    println!("baseline: expl={expl:.3e} ev={ev:?} ip={ip_strategy:?}");
    assert!(expl < 1e-2, "baseline exploitability too large: {expl}");
    assert_eq!(ip_strategy.len(), 2);
}

#[test]
fn locked_node_reports_locked_frequencies() {
    let (expl, _ev, ip_strategy) = solve_with_ip_lock(Some((0.25, 0.75)));
    println!("locked 0.25/0.75: expl={expl:.3e} ip={ip_strategy:?}");
    assert!(
        (ip_strategy[FOLD] - 0.25).abs() < 1e-3,
        "fold freq should equal the lock, got {}",
        ip_strategy[FOLD]
    );
    assert!(
        (ip_strategy[CALL] - 0.75).abs() < 1e-3,
        "call freq should equal the lock, got {}",
        ip_strategy[CALL]
    );
}

#[test]
fn restricted_exploitability_converges_with_locks() {
    // If the BR used for exploitability ignored the lock, this could never reach ~0
    // (the locked node would be treated as freely deviating). It must converge.
    let (expl, _ev, _ip) = solve_with_ip_lock(Some((0.5, 0.5)));
    println!("locked 0.5/0.5: expl={expl:.3e}");
    assert!(
        expl < 1e-2,
        "restricted exploitability did not converge: {expl}"
    );
    // and a re-computation of the same quantity should agree
    let mut game = build_game();
    game.allocate_memory(false);
    game.play(1);
    game.lock_current_strategy(&[0.5, 0.5]);
    game.back_to_root();
    solve(&mut game, 2000, 1e-4, false);
    let recomputed = compute_exploitability(&game);
    println!("recomputed: {recomputed:.3e}");
    assert!(recomputed < 1e-2, "recomputed: {recomputed}");
}

#[test]
fn lock_orientation_shifts_oop_ev_monotonically() {
    let (_e0, ev_baseline, _s0) = solve_with_ip_lock(None);
    let (_e1, ev_fold_heavy, s1) = solve_with_ip_lock(Some((0.9, 0.1)));
    let (_e2, ev_call_heavy, s2) = solve_with_ip_lock(Some((0.1, 0.9)));

    println!("baseline   ev={ev_baseline:?}");
    println!("fold-heavy ev={ev_fold_heavy:?} ip={s1:?}");
    println!("call-heavy ev={ev_call_heavy:?} ip={s2:?}");

    // IP folds too much -> OOP's EV must improve vs. baseline
    assert!(
        ev_fold_heavy[0] > ev_baseline[0] + 1e-3,
        "IP folding more must raise OOP EV: {:?} vs {:?}",
        ev_fold_heavy,
        ev_baseline
    );
    // Hand-computed reference values for this toy tree (pot 20, stack 10, OOP={AA,QQ}, IP={KK},
    // reported ev = net_stack_change - pot/2, averaged over OOP's two equally weighted hands):
    //   IP locked to fold with freq f: AA bet = f*10 + (1-f)*20, QQ bet = f*10 + (1-f)*(-20),
    //   and QQ's alternative (check) is worth -10. Hence
    //     f = 2/3 (equilibrium, MDF) -> 3.3333,  f = 0.9 -> 9.0,  f = 0.1 -> 4.5,
    //     f = 1.0 -> 10.0 (OOP takes the whole pot),  f = 0.0 -> 5.0.
    assert!(
        (ev_baseline[0] - 3.33333).abs() < 5e-3,
        "baseline ev expected 3.33333, got {:?}",
        ev_baseline
    );
    // A *locked* (non-equilibrium) opponent can only be exploited further: the free player's BR
    // value must be >= its equilibrium value, i.e. the equilibrium strategy is least exploitable.
    assert!(
        ev_fold_heavy[0] > ev_baseline[0] + 1e-3,
        "IP folding more must raise OOP BR EV: {:?} vs {:?}",
        ev_fold_heavy,
        ev_baseline
    );
    assert!(
        ev_call_heavy[0] > ev_baseline[0] + 1e-3,
        "IP calling more must also raise OOP BR EV: {:?} vs {:?}",
        ev_call_heavy,
        ev_baseline
    );
    // Symmetrically, the locked player can never beat its own equilibrium payoff.
    assert!(
        ev_fold_heavy[1] <= ev_baseline[1] + 1e-3,
        "locked IP should not gain: {:?} vs {:?}",
        ev_fold_heavy,
        ev_baseline
    );
    assert!(
        ev_call_heavy[1] <= ev_baseline[1] + 1e-3,
        "locked IP should not gain: {:?} vs {:?}",
        ev_call_heavy,
        ev_baseline
    );
    // Analytic pins: these validate the *orientation* of the lock, not merely its existence.
    assert!(
        (ev_fold_heavy[0] - 9.0).abs() < 5e-3,
        "fold-heavy ev expected 9.0, got {:?}",
        ev_fold_heavy
    );
    assert!(
        (ev_call_heavy[0] - 4.5).abs() < 5e-3,
        "call-heavy ev expected 4.5, got {:?}",
        ev_call_heavy
    );
    assert!(ev_fold_heavy[0] > ev_call_heavy[0]);
}

#[test]
fn pure_locks_match_exact_game_theory() {
    // f = 1.0: IP always folds to the all-in -> OOP wins the pot with both hands -> +10.
    let (_e, ev, s) = solve_with_ip_lock(Some((1.0, 0.0)));
    println!("always-fold ev={ev:?} ip={s:?}");
    assert!(s[0] > 0.999, "lock must be reported as a pure fold, got {s:?}");
    assert!((ev[0] - 10.0).abs() < 1e-2, "expected +10, got {:?}", ev);

    // f = 0.0: IP always calls -> AA wins the 40 pot (+20), QQ check/folds (-10) -> +5.
    let (_e, ev, s) = solve_with_ip_lock(Some((0.0, 1.0)));
    println!("always-call ev={ev:?} ip={s:?}");
    assert!(s[1] > 0.999, "lock must be reported as a pure call, got {s:?}");
    assert!((ev[0] - 5.0).abs() < 1e-2, "expected +5, got {:?}", ev);
}


/// Solves with an optional IP lock and reports both exploitability definitions.
///
/// `restricted`   = `compute_exploitability()`: best response constrained by the locks.
/// `unrestricted` = `compute_exploitability_unrestricted()`: best response free to deviate
///                  everywhere, i.e. how far the locked profile is from an equilibrium.
fn solve_and_measure(lock: Option<(f32, f32)>) -> (f32, f32, [f32; 2]) {
    let mut game = build_game();
    game.allocate_memory(false);
    if let Some((fold, call)) = lock {
        game.play(1);
        game.lock_current_strategy(&[fold, call]);
        game.back_to_root();
    }
    solve(&mut game, 2000, 1e-4, false);
    game.cache_normalized_weights();
    let restricted = compute_exploitability(&game);
    let unrestricted = compute_exploitability_unrestricted(&game);
    let mes_free = compute_mes_ev_unrestricted(&game);
    let ev = compute_current_ev(&game);
    println!(
        "lock={lock:?} restricted={restricted:.3e} unrestricted={unrestricted:.4} mes_free={mes_free:?} ev={ev:?}"
    );
    (restricted, unrestricted, ev)
}

#[test]
fn unrestricted_exploitability_measures_the_cost_of_locks() {
    // 1) Without locks both definitions are the same computation and must agree exactly.
    let (r0, u0, _ev0) = solve_and_measure(None);
    assert!(
        (r0 - u0).abs() < 1e-6,
        "without locks the two definitions must agree: {r0} vs {u0}"
    );
    assert!(u0 < 1e-2, "and the base game must still be solved: {u0}");

    let (r1, u1, ev1) = solve_and_measure(Some((1.0, 0.0)));
    let (r2, u2, ev2) = solve_and_measure(Some((0.5, 0.5)));

    // 2) With locks the lock-aware value collapses to ~0 (the solver stops), while the
    //    unrestricted one stays clearly positive: the profile is not an equilibrium.
    assert!(r1 < 1e-2 && r2 < 1e-2, "restricted: {r1} {r2}");
    assert!(u1 > 0.5 && u2 > 0.5, "unrestricted: {u1} {u2}");
    assert!(u1 > r1 + 0.5 && u2 > r2 + 0.5);

    // 3) The unrestricted BR set is a superset, and more extreme locks are more exploitable.
    assert!(u1 >= r1 - 1e-6 && u2 >= r2 - 1e-6);
    assert!(u1 > u2, "more extreme locks must be more exploitable: {u1} vs {u2}");

    // 4) Consistency with the profile EV: a locked profile that hands OOP +10 while IP
    //    would rather call is exploitable, so the unrestricted value must exceed it.
    assert!((ev1[0] - 10.0).abs() < 1e-2, "pure fold ev: {ev1:?}");
    assert!(u1 > ev1[0] * 0.2, "sanity on magnitudes: {u1} vs {ev1:?}");
    assert!((ev2[0] - 5.0).abs() < 1e-2, "50/50 ev: {ev2:?}");
}

#[test]
fn equilibrium_frequency_lock_keeps_ev_but_can_still_be_exploitable() {
    // Measured behavior: locking a node changes *which* equilibrium the solver converges to.
    // Pinning the unlocked solve's frequency at the IP node keeps the EV essentially unchanged,
    // but the delivered profile is still exploitable by a free best response. The restricted
    // metric cannot see this because it holds the locked node fixed for the best response too.
    let (expl_base, ev_base, ip_strategy) = solve_with_ip_lock(None);

    let mut game = build_game();
    game.allocate_memory(false);
    game.play(1);
    game.lock_current_strategy(&ip_strategy);
    game.back_to_root();
    let expl_locked = solve(&mut game, 2000, 1e-4, false);
    game.cache_normalized_weights();
    let ev_locked = compute_current_ev(&game);
    let restricted = compute_exploitability(&game);
    let free = compute_exploitability_unrestricted(&game);

    println!("baseline expl={expl_base:.3e} ev={ev_base:?} ip={ip_strategy:?}");
    println!("equilibrium-frequency lock expl={expl_locked:.3e} restr={restricted:.3e} free={free:.3e} ev={ev_locked:?}");

    // 1. a faithful lock barely moves the EV of the locked profile
    assert!(
        (ev_locked[0] - ev_base[0]).abs() < 1e-2,
        "EV changed by a faithful lock: {:?} vs {:?}",
        ev_locked,
        ev_base
    );
    // 2. the restricted metric keeps converging (the solver optimizes *within* the lock)
    assert!(
        restricted < 1e-2,
        "restricted metric did not converge: {restricted}"
    );
    // 3. only the unrestricted metric reveals the cost of locking
    assert!(
        free > restricted,
        "free({free}) must dominate restricted({restricted})"
    );
    // 4. regression fixture for this toy tree: measured free exploitability ~1.52
    assert!(
        free > 0.5,
        "expected a large unrestricted exploitability, got {free}"
    );
}

#[test]
fn unrestricted_equals_restricted_without_locks() {
    // Structural identity: with no locks at all the two metrics must agree exactly.
    let mut game = build_game();
    game.allocate_memory(false);
    solve(&mut game, 2000, 1e-4, false);
    let restricted = compute_exploitability(&game);
    let free = compute_exploitability_unrestricted(&game);
    println!("no locks: restricted={restricted:.6e} free={free:.6e}");
    assert!(
        (restricted - free).abs() < 1e-6,
        "metrics differ without locks: {restricted} vs {free}"
    );
}

// ---------------------------------------------------------------------------
// Phase 3 (3A): soft locks ("restricted response") - the locked frequencies are
// blended with the solved ones by a weight in `[0, 1]`.
//
// Semantics under test (see `lock_current_strategy_with_weight`):
//   realized(node) = weight * target + (1 - weight) * solved
// - weight = 1.0 is a hard lock (bit-for-bit the old behavior)
// - weight = 0.0 leaves the node free (the result must match the unlocked solve)
// - 0 < weight < 1: only the residual frequency `1 - weight` is optimized by the
//   solver, and the realized (blended) strategy is what the node plays, what is
//   reported by `strategy()`, and what both exploitability metrics measure.
// ---------------------------------------------------------------------------

/// Solves with an optional *soft* IP lock and reports both metrics, the root EV
/// and the (blend-aware) strategy reported at the locked node.
fn solve_soft(lock: Option<(f32, f32)>, weight: f32) -> (f32, f32, [f32; 2], Vec<f32>) {
    let mut game = build_game();
    game.allocate_memory(false);
    if let Some(strategy) = lock {
        game.play(1);
        game.lock_current_strategy_with_weight(&[strategy.0, strategy.1], weight);
        // the weight must be stored (clamped) and readable at the locked node
        let stored = game
            .current_locking_weight()
            .expect("a locked node must report a weight");
        assert!(
            (stored - weight.clamp(0.0, 1.0)).abs() < 1e-6,
            "weight {weight} must be stored clamped, got {stored}"
        );
        game.back_to_root();
    }
    let restricted = solve(&mut game, 2000, 1e-4, false);
    game.cache_normalized_weights();
    let free = compute_exploitability_unrestricted(&game);
    let ev = compute_current_ev(&game);
    game.play(1);
    let reported = game.strategy();
    game.back_to_root();
    println!(
        "lock={lock:?} weight={weight} restricted={restricted:.3e} free={free:.4} ev={ev:?} ip={reported:?}"
    );
    (restricted, free, ev, reported)
}

#[test]
fn soft_lock_weight_one_matches_hard_lock() {
    // weight = 1.0 must be indistinguishable from the (hard) lock API.
    for lock in [(0.25, 0.75), (1.0, 0.0), (0.9, 0.1)] {
        let (r_hard, ev_hard, s_hard) = solve_with_ip_lock(Some(lock));
        let (r_soft, _f_soft, ev_soft, s_soft) = solve_soft(Some(lock), 1.0);
        println!("hard {lock:?}: ev={ev_hard:?} ip={s_hard:?}");
        println!("soft {lock:?}: ev={ev_soft:?} ip={s_soft:?}");
        assert!(
            (r_hard - r_soft).abs() < 1e-6,
            "restricted exploitability differs: {r_hard} vs {r_soft}"
        );
        assert!(
            (ev_hard[0] - ev_soft[0]).abs() < 1e-4 && (ev_hard[1] - ev_soft[1]).abs() < 1e-4,
            "EV differs for weight 1.0: {ev_hard:?} vs {ev_soft:?}"
        );
        assert_eq!(s_hard.len(), s_soft.len());
        for (a, b) in s_hard.iter().zip(s_soft.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "reported strategy differs for weight 1.0: {s_hard:?} vs {s_soft:?}"
            );
        }
    }
}

#[test]
fn soft_lock_weight_zero_matches_unlocked() {
    // weight = 0.0 must reproduce the unlocked solve exactly (the node stays marked as
    // locked, but its realized strategy is 100% the solved one).
    let (r0, ev0, s0) = solve_with_ip_lock(None);
    let (r1, _f1, ev1, s1) = solve_soft(Some((1.0, 0.0)), 0.0);
    println!("unlocked: ev={ev0:?} ip={s0:?}");
    println!("weight 0: ev={ev1:?} ip={s1:?}");
    assert!(
        (ev0[0] - ev1[0]).abs() < 1e-3 && (ev0[1] - ev1[1]).abs() < 1e-3,
        "EV must equal the unlocked EV: {ev0:?} vs {ev1:?}"
    );
    for (a, b) in s0.iter().zip(s1.iter()) {
        assert!(
            (a - b).abs() < 1e-3,
            "weight 0 must report the solved strategy: {s0:?} vs {s1:?}"
        );
    }
    assert!(
        (r0 - r1).abs() < 1e-6,
        "restricted metric must be unbiased: {r0} vs {r1}"
    );
}

#[test]
fn soft_lock_weight_is_clamped() {
    let (_r, _f, ev_one, s_one) = solve_soft(Some((0.9, 0.1)), 1.0);
    let (_r2, _f2, ev_over, s_over) = solve_soft(Some((0.9, 0.1)), 2.5);
    let (_r0, _f0, ev_zero, s_zero) = solve_soft(Some((0.9, 0.1)), 0.0);
    let (_r3, _f3, ev_neg, s_neg) = solve_soft(Some((0.9, 0.1)), -3.0);
    println!("weight>1: ev={ev_over:?} ip={s_over:?} (vs 1.0: {ev_one:?} {s_one:?})");
    println!("weight<0: ev={ev_neg:?} ip={s_neg:?} (vs 0.0: {ev_zero:?} {s_zero:?})");
    for (a, b) in s_one.iter().zip(s_over.iter()) {
        assert!((a - b).abs() < 1e-6, "weight > 1 must clamp to 1: {s_one:?} vs {s_over:?}");
    }
    assert!((ev_one[0] - ev_over[0]).abs() < 1e-4);
    for (a, b) in s_zero.iter().zip(s_neg.iter()) {
        assert!((a - b).abs() < 1e-3, "weight < 0 must clamp to 0: {s_zero:?} vs {s_neg:?}");
    }
    assert!((ev_zero[0] - ev_neg[0]).abs() < 1e-3);
}

#[test]
fn soft_lock_reported_strategy_is_the_blend() {
    // Structural invariant: what the node reports is `weight * target + (1 - weight) * solved`,
    // so the residual solved part recovered from the report must be a valid distribution.
    let target = (0.9f32, 0.1f32);
    for weight in [0.25f32, 0.5, 0.75] {
        let (_r, _f, _ev, s) = solve_soft(Some(target), weight);
        let recovered_fold = (s[FOLD] - weight * target.0) / (1.0 - weight);
        let recovered_call = (s[CALL] - weight * target.1) / (1.0 - weight);
        println!("weight={weight} report={s:?} recovered={recovered_fold:.4}/{recovered_call:.4}");
        assert!((s[FOLD] + s[CALL] - 1.0).abs() < 1e-4, "not a distribution: {s:?}");
        assert!(
            (-1e-4..=1.0 + 1e-4).contains(&recovered_fold)
                && (-1e-4..=1.0 + 1e-4).contains(&recovered_call),
            "the residual part is not a valid distribution: {recovered_fold}/{recovered_call}"
        );
        assert!(
            (recovered_fold + recovered_call - 1.0).abs() < 1e-3,
            "the residual part must sum to one: {recovered_fold}/{recovered_call}"
        );
    }
}

#[test]
fn non_binding_soft_lock_is_less_exploitable_than_a_hard_lock() {
    // A hard lock to a non-equilibrium frequency (0.5/0.5) pins the node and its
    // unrestricted exploitability stays large (the profile is not an equilibrium).
    // The same target as a *soft* lock leaves `[0.25, 0.75]` of fold frequencies free,
    // which still contains the equilibrium frequency (~0.667), so the solver can return
    // to an (almost) unexploitable profile: the free metric collapses.
    let (_r_hard, free_hard, ev_hard, s_hard) = solve_soft(Some((0.5, 0.5)), 1.0);
    let (_r_soft, free_soft, ev_soft, s_soft) = solve_soft(Some((0.5, 0.5)), 0.5);
    println!("hard 0.5/0.5   free={free_hard:.4} ev={ev_hard:?} ip={s_hard:?}");
    println!("soft 0.5 (w=.5) free={free_soft:.4} ev={ev_soft:?} ip={s_soft:?}");
    assert!(free_hard > 0.5, "the hard lock must remain exploitable: {free_hard}");
    assert!(
        free_soft < 1e-2,
        "a non-binding soft lock must return to an unexploitable profile, got {free_soft}"
    );
    assert!(
        free_soft < free_hard,
        "the soft lock must be less exploitable: {free_soft} vs {free_hard}"
    );
    // and the reported strategy is a genuine blend of the target and the solved part
    let recovered = (s_soft[FOLD] - 0.5 * 0.5) / 0.5;
    assert!(
        (0.0..=1.0).contains(&recovered),
        "recovered residual out of range: {recovered}"
    );
}

#[test]
fn binding_soft_lock_equals_the_equivalent_hard_lock() {
    // Target "always fold" with weight 0.8 restricts the node to `[0.8, 1.0]`; the solver
    // minimizes OOP's BR value, which is increasing in the fold frequency here, so the
    // optimum sits on the boundary 0.8 - exactly the profile of a hard lock at 0.8.
    let (_r_hard, free_hard, ev_hard, s_hard) = solve_soft(Some((1.0, 0.0)), 1.0);
    let (_r_ref, free_ref, ev_ref, s_ref) = solve_soft(Some((0.8, 0.2)), 1.0); // hard lock at 0.8
    let (_r_soft, free_soft, ev_soft, s_soft) = solve_soft(Some((1.0, 0.0)), 0.8);
    println!("hard 1.0 (w=1) free={free_hard:.4} ev={ev_hard:?} ip={s_hard:?}");
    println!("hard 0.8 (w=1) ev={ev_ref:?} ip={s_ref:?}");
    println!("soft fold-all (w=.8) free={free_soft:.4} ev={ev_soft:?} ip={s_soft:?}");
    // the realized fold frequency of the binding soft lock is the constraint boundary
    assert!(
        (s_soft[FOLD] - 0.8).abs() < 5e-3,
        "expected the boundary fold frequency 0.8, got {s_soft:?}"
    );
    // and hence the delivered profile (EV, exploitability) matches the hard lock at 0.8
    assert!(
        (ev_soft[0] - ev_ref[0]).abs() < 5e-3,
        "EV must match the equivalent hard lock: {ev_soft:?} vs {ev_ref:?}"
    );
    assert!(
        (free_soft - free_ref).abs() < 5e-3,
        "free exploitability must match the equivalent hard lock: {free_soft}"
    );
    // a binding soft lock is still exploitable, but strictly less than the hard lock
    assert!(free_soft > 1e-3, "a binding soft lock must cost something: {free_soft}");
    assert!(
        free_soft < free_hard,
        "the soft lock must be less exploitable than the hard one: {free_soft} vs {free_hard}"
    );
}

#[test]
fn lock_bookkeeping_tracks_the_weight() {
    let mut game = build_game();
    game.allocate_memory(false);

    // not locked
    game.play(1);
    assert!(game.current_locking_strategy().is_none());
    assert!(game.current_locking_weight().is_none());

    // locked with a soft weight
    game.lock_current_strategy_with_weight(&[0.3, 0.7], 0.4);
    let stored = game.current_locking_strategy().unwrap();
    assert!((stored[0] - 0.3).abs() < 1e-6 && (stored[1] - 0.7).abs() < 1e-6);
    assert_eq!(game.current_locking_weight(), Some(0.4));

    // unlocking must drop the weight, not only the strategy
    game.unlock_current_strategy();
    assert!(game.current_locking_strategy().is_none());
    assert!(
        game.current_locking_weight().is_none(),
        "unlock must clear the weight"
    );

    // locking again, then clearing everything
    game.lock_current_strategy_with_weight(&[0.9, 0.1], 0.5);
    assert_eq!(game.current_locking_weight(), Some(0.5));
    game.clear_locking_strategy();
    assert!(
        game.current_locking_weight().is_none(),
        "clear must drop all weights"
    );
    game.back_to_root();
}
