# Node Locking — engine notes

Branch: `nodelock` (local fork of `postflop-solver`, consumed by the *Desktop Postflop*
`nodelock` branch). Not upstreamed; `src-tauri/Cargo.toml` points at this clone by path.

Written 2026-10-03. Companion document (UI / app side): `desktop-postflop/docs/NODELOCK.md`.

---

## 1. Purpose

Pin the strategy of a chosen decision node — typically the *villain* node the hero is facing —
to a chosen (possibly non-equilibrium) frequency mix, then re-solve. The result is the optimal
*restricted response* of the other player to that fixed strategy, plus a measure of how far the
locked profile is from an equilibrium.

Typical question answered: *"if villain keeps betting 60% / checking 40% here, how do I play?"*

---

## 2. Terminology

| Term | Meaning |
|---|---|
| **node lock** | A node whose strategy is pinned for the whole solve. |
| **locking strategy** | Target frequencies stored for a locked node, flat layout `index = action * num_hands + hand`. Length = `#actions * #hands` = `expected_strategy_len()`. A private hand whose entries are all `< 0.0`/`<= 0.0` is **not locked**; positive entries are normalized to sum to 1. |
| **hard lock** | `weight = 1.0`. The node plays exactly the target frequencies; CFR does not optimize it. |
| **soft lock / RRN** | `0.0 < weight < 1.0`. On every iteration the node plays `weight * target + (1 - weight) * current`, i.e. the target is approached gradually and the residual frequency `1 - weight` stays free for the solver ("restricted Nash response" refinement). `weight <= 0` is equivalent to unlocked; the weight is clamped to `[0, 1]`. |
| **locking-aware strategy** | The single helper `apply_locking_strategy()` used by the solver, the result interpreter and the EV/exploitability code, so that *solved = displayed = evaluated*. |
| **restricted exploitability** | `compute_exploitability()`: best response is **constrained** at locked nodes. Converges to ~0 whenever the solver has converged *within* the restrictions. This is the stopping criterion. |
| **unrestricted ("real") exploitability** | `compute_exploitability_unrestricted()`: best response is **free everywhere**. This is the true exploitability of the delivered profile and does *not* have to approach 0. |
| **cost of adjustment** | EV difference at the root between the locked profile and the baseline (unlocked) solution. |
| **frequency target** | A node-level frequency per action which the app expands into per-hand locking entries proportionally to the current strategy. |
| **sensitivity sweep** | Repeated full re-solves over a grid of one lock parameter (soft-lock weight, or one action's node frequency), producing an EV / exploitability curve. |

---

## 3. Public API

```rust
// booking (valid after `allocate_memory()` and before the game is solved; panics otherwise)
PostFlopGame::lock_current_strategy(&mut self, strategy: &[f32]);
PostFlopGame::lock_current_strategy_with_weight(&mut self, strategy: &[f32], weight: f32);
PostFlopGame::unlock_current_strategy(&mut self);
PostFlopGame::clear_locking_strategy(&mut self);
PostFlopGame::current_locking_strategy(&self) -> Option<Vec<f32>>;   // stored targets
PostFlopGame::current_locking_weight(&self) -> Option<f32>;          // None => hard lock
PostFlopGame::expected_strategy_len(&self) -> usize;                 // #actions * #hands

// metrics
compute_exploitability(&game)             -> f32          // restricted (locks respected)
compute_exploitability_unrestricted(&game) -> f32          // real exploitability
compute_mes_ev(&game)                      -> [f32; 2]     // restricted best response
compute_mes_ev_unrestricted(&game)         -> [f32; 2]     // free best response
```

Notes
* `solve()` / `solve_step()` always solve the whole tree from `game.root()`; the current node
  only selects *which* node is being read/locked. The app therefore navigates with
  `apply_history()` and locks the node it is standing on.
* `strategy()` is lock-aware: at a locked node it returns the (blended, if soft) locked
  frequencies, not the raw CFR average.

---

## 4. How it works (code map)

* `src/game/interpreter.rs`
  * `locking_strategy: BTreeMap<usize, Vec<f32>>` and `locking_weight: BTreeMap<usize, f32>`
    keyed by node index; exposed through the `Game` trait (`locking_strategy()`,
    `locking_weight(node)`).
  * `strategy()` / `cfreach_actions()` / terminal handling go through
    `apply_stored_locking_strategy()`.
  * **Rebuilding the tree invalidates every lock** (node indices are recreated) — the maps are
    cleared on rebuild.
* `src/utility.rs`
  * `apply_locking_strategy(dst, locking, weight)`: `weight * target + (1 - weight) * dst` for
    every entry of a locked private hand; `weight >= 1` = hard pin, `weight <= 0` = no-op.
  * `apply_stored_locking_strategy(dst, locking, weight)`: used on strategies that came out of
    the solver. The solver already mixes a soft lock into every strategy it accumulates, so a
    stored strategy is re-pinned **only for hard locks** (`weight >= 1.0`). Re-mixing would
    apply the blend twice and pull every reported frequency towards the target
    (`weight -> weight * (2 - weight)`) — this was a real bug, fixed on this branch.
  * `compute_mes_ev_inner(game, ignore_locking)`: single implementation of the restricted
    (`ignore_locking = false`) and unrestricted (`ignore_locking = true`) best response. At a
    hard-locked node the best response actions are forced to the lock; at a soft-locked node the
    hard best response is blended with the locked mixture by the weight.
* `src/solver.rs`: the regret update skips locked private hands, so a locked node's strategy does
  not drift; the average strategy is the locked one.
* `src/interface.rs`, `src/game/mod.rs`, `src/action_tree.rs`: trait surface and helpers
  (`locking_weight`, `expected_strategy_len`, stale-lock cleanup on rebuild).
* Serialization: the **locking strategy** is serialized with the game (bincode), the
  **weight is not** — it is a transient solve option and must be restored after unserializing.

---

## 5. Metric semantics — documented known behaviour

Locking a node **changes which equilibrium the solver converges to**, so a "faithful" lock (one
that pins the node to the frequencies it has in the unlocked solution) still produces a profile
that a free best response can exploit. Both numbers are legitimate, they answer different
questions:

* restricted (`compute_exploitability`) — "did the solve converge inside the locks?"
* unrestricted (`compute_exploitability_unrestricted`) — "how exploitable is the profile I am
  actually looking at?"

Invariants (asserted in `tests/nodelock_verify.rs`):

* `unrestricted >= restricted`, with equality when no node is locked.
* `restricted < 1e-2` after a converged solve — even with locks present.
* Pure locks reproduce game theory exactly: pinning the IP node to *always fold* after an OOP
  all-in gives OOP root EV `+10` (pot 20), *always call* gives `+5`; a 50/50 lock gives `+5`.
* More extreme locks are more exploitable: `unrestricted(pure fold) > unrestricted(50/50) > 0`.
* Soft locks with `weight = 1` are identical to hard locks, `weight = 0` identical to unlocked,
  values outside `[0, 1]` are clamped.

Measured example (river subgame: `AsAh,QsQh` vs `KsKh`, board `2s3h4d 6c 7c`, pot 20, stack 10,
equilibrium-frequency lock): an earlier run of
`equilibrium_frequency_lock_keeps_ev_but_can_still_be_exploitable` printed
`restricted ≈ 7.8e-5` while `unrestricted ≈ 1.52` chips — i.e. a converged-looking, EV-preserving
lock whose profile is still exploitable. This is the documented, accepted behaviour (it is not a
bug); the stopping criterion must therefore use the **restricted** value, and the UI must never
present the restricted value alone as "exploitability of the displayed strategy".

---

## 6. Verification

```bash
cargo +nightly test --offline                      # full suite (70 tests as of 2026-10-03)
cargo +nightly test --offline --test nodelock_verify
```

`tests/nodelock_verify.rs` (16 tests) covers:

| Test | Asserts |
|---|---|
| `baseline_has_no_lock_and_converges` | unlocked solve: no lock booked, `exploitability < 1e-2` |
| `locked_node_reports_locked_frequencies` | `strategy()` returns exactly the locked mix |
| `restricted_exploitability_converges_with_locks` | restricted metric still converges with locks |
| `lock_orientation_shifts_oop_ev_monotonically` | fold-heavy vs call-heavy locks move OOP EV monotonically |
| `pure_locks_match_exact_game_theory` | pure locks match hand-computed EVs (`+10` / `+5`) |
| `unrestricted_exploitability_measures_the_cost_of_locks` | `unrestricted > 0.5`, `> restricted + 0.5`, ordering by extremity |
| `equilibrium_frequency_lock_keeps_ev_but_can_still_be_exploitable` | EV preserved, restricted ~0, unrestricted large |
| `unrestricted_equals_restricted_without_locks` | equal with no locks |
| `soft_lock_weight_one_matches_hard_lock` | `weight = 1` ≡ hard lock |
| `soft_lock_weight_zero_matches_unlocked` | `weight = 0` ≡ unlocked |
| `soft_lock_weight_is_clamped` | weights outside `[0, 1]` clamp |
| `soft_lock_reported_strategy_is_the_blend` | reported strategy is `weight * target + (1 - weight) * solved` and a distribution |
| `non_binding_soft_lock_is_less_exploitable_than_a_hard_lock` | residual freedom reduces exploitability |
| `binding_soft_lock_equals_the_equivalent_hard_lock` | a binding soft lock matches the equivalent hard lock, and costs EV |
| `lock_bookkeeping_tracks_the_weight` | `current_locking_strategy/weight` bookkeeping, incl. unlock |

Performance (release, same machine, 100 solver steps on a turn subgame; harness not committed,
lives under `temp/perf_new` (fork) / `temp/perf_orig` (upstream)):

| Build | ms / step |
|---|---|
| upstream engine | 1.674 |
| this branch | 1.535 |

⇒ locking costs nothing measurable; the 5-10x slowdown originally observed in the app was a
`tauri dev` build-profile issue, not an engine regression (see the app document, §7).

---

## 7. Reproduction

```bash
# engine
git -C C:/Users/Tony/Projects/postflop-solver checkout nodelock
cargo +nightly test --offline                      # Cargo.lock pins bincode 2.0.0-rc.3 (offline builds)
cargo +nightly run --release --example <name>      # examples/build_tree.rs, examples/read_solution.rs

# nightly is required: the crate uses #![feature(allocator_api)] for `custom-alloc`.
```

`--offline` only works with the pinned `Cargo.lock`; regenerating it may fetch newer
pre-release dependencies.

---

## 8. Limitations

* Locks may only be set between `allocate_memory()` and the first solve step; the engine `panic!`s
  otherwise. The Tauri layer pre-checks this and returns a message instead of crashing.
* A tree rebuild drops all locks (node indices change).
* `load_from_file` / unserializing restores locking strategies but **not** weights.
* Unrestricted exploitability is `O(tree)` per call and, unlike the restricted one, is not used
  for convergence — call it sparingly (the app refreshes it once per solve / sweep point).
* The engine keeps a single game instance in the app; every lock edit requires a full re-solve.
