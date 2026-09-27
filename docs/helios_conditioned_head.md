# Helios seed 1: largest measured policy defect and repair

The prior pilot did not clearly outperform RPA. Multi-seed replication was
stopped to investigate the policy limitation before fresh seed-1 training.
Seed 2's unfinished artifacts are preserved; seeds 3–5 were not trained.

## Root cause

The ST-GNN rack scoring head is a single linear layer applied to
`[rack_embedding, global_embedding, job_context, full_counterfactual_vector]`.
The global/context terms are identical across racks. Therefore
`logit[r] = w_rack · embedding[r] + shared_offset(context)`, and that offset
cancels in softmax. Conditional on legal racks, placement probabilities cannot
respond to job size, network context or action counterfactual utilities.
The DEFER head is separate, but it does not restore context-dependent relative
ranking between placement anchors. The same structural defect affects A2.

This is an expressivity defect, not an insufficient number of training seeds.
A gradient check gives context sensitivity below 1e-9; replacing the entire
context changes placement probabilities by only about 3e-8 (float32 roundoff).
With identical rack telemetry and two jobs requiring opposite anchors, the old
actor still predicts the same anchor after 80 optimizer steps.

## Attribution on training data only

Frozen seed 1 final weights, hours 1000–1010, 50% load (358 jobs), identical
reward weights/simulator and pointwise decoder. Oracle variants are diagnostic
ablations; they are not the repaired production decoder.

| Variant | Reward | Rack samples | Completed | Preempt | Wait cost |
|---|---:|---:|---:|---:|---:|
| RPA | 19.79567 | 0 | 357 | 10 | 376 |
| Learned | 19.76285 | 439 | 357 | 9 | 366 |
| A4 best counterfactual | 19.81057 | 256 | 357 | 8 | 365 |
| No optional DEFER | 19.76285 | 439 | 357 | 9 | 366 |
| A2/A3/A1 best counterfactual | 19.76285 | 439 | 357 | 9 | 366 |
| All counterfactual oracles | 19.81057 | 256 | 357 | 8 | 365 |

A4 is the largest measured policy bottleneck among these interventions. Removing
optional DEFER cannot help this replay: there are none. Relief policies already
agree with their counterfactual targets here. A4 oracle improves reward by
0.04771 and reduces violation samples by 41.7%, but still has 256 violations.
These results identify the direction of repair; they do not guarantee a trained
neural policy will beat RPA on independent validation.

## Repair

In a new isolated fork, each rack's counterfactual utility is aligned with that
rack's score input instead of broadcasting the entire utility vector. A small
nonlinear score head combines the rack embedding with the global job context,
allowing the context to change relative rack preference. DEFER gets its own
utility and global context. A2 uses the corresponding rack risk scalar.
The observation layout and actor outputs remain the same; the internal model
architecture changes, requiring a fresh training run.

The class keeps a legacy default (`action_context_dim=0`); corrected A2/A4
instances explicitly enable aligned action context. No hard-coded greedy action
override or safety shield is introduced. Rewards, workload, power assignment,
host allocator, baselines, horizons and both decoders are unchanged.

Regression evidence: two behavioral tests fail on the old actor and pass after
repair. The model can learn opposite anchors under identical telemetry; global
job context has nonzero influence on relative rack probabilities. Masked actions
and backward gradients remain valid. 14 tests pass: 3 new actor tests, 2 entropy,
2 CF-zero-loss, 3 launcher/isolation and 4 training-resume tests.

## New pilot and isolation

Source: `/teamspace/studios/this_studio/pa4h_helios_conditioned_20260927`.
Session: `helios_conditioned_seed1`.
Output: `outputs/helios_conditioned_300_seed1/` inside that fork.
Fresh seed 1, 300 episodes, train 1000–1010 at 50%; primary validation 1010–1020
at 5%, stress at 50%, both decoders. Holdout 1020–1030 stays reserved.
The runner snapshots/verifies and evaluates after training. No follow-on seed
queue is active. After completion, compare against the old pilot on the same
protocol before replication. Preserve unfavorable results as well.

Python/Java source, dependencies and binaries are copied; modified model/trainer
files live only in the new fork. The curated repository ships these changes as
`reproduction/helios_conditioned.patch`, applied by `reproduction/prepare.py`.
Local Alibaba model/trainer files and server `pa4h_v5_run` are not edited.
Existing Helios checkpoints and source lineage are preserved. The new source
manifest includes the model and trainer hashes, and the protocol records the
architecture change.

Raw train-only attribution is in `results/helios_placement_attribution.json`.
The diagnostic script `reproduction/diagnose_placement.py` requires the prior
pilot's complete checkpoint and a running matching bridge. It is an oracle
diagnostic, not a production evaluation or the corrected pilot's result.
