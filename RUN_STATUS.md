# Run status — 2026-09-27

## Alibaba: five completed historical seeds

300 episodes per seed. Results and exact recorded settings are in
`results/alibaba_5seeds.csv` and `reproduction/alibaba_seeds.json`.
The archive in `reproduction/alibaba_source.sha256` records historical source
hashes; some historical launch scripts are intentionally not shipped.

**Do not run all five seeds against one trainer version.**

| Seeds | Trainer SHA-256 | Source preparation |
| --- | --- | --- |
| 1–2 | `d10309428215ec4e23066c7d19af016ae7f38edf0f2fd9143190c2c602b2a761` | Restore using `reproduction/alibaba_pre_entropy.patch` |
| 3–5 | `ba7c4dd8f262cc722048a442f1e2b08c8542cca792b3ee722c90b8018e0309b4` | Base trainer committed in `python/` |

The old trainer was recovered from local `v5_resume.zip`; this ZIP was not
available on the training server. The small patch now restores the exact old
trainer without committing that archive. `reproduction/prepare.py` selects and
verifies the trainer for each seed and creates separate source directories.
The old generic launchers do not implement this selection.

Seed 3 historically trained with the old source until checkpoint 224 and resumed
with the entropy hotfix. A fresh seed 3 using the hotfix from episode 0 is a
new replication, not a bit-identical reconstruction of that historical run.
The historical seed 1–2 trainer retains the masked-entropy numerical issue;
restoring its hash does not guarantee a stable fresh replay on another runtime.
Keep historical results and fresh replications separate.

Server artifacts under `/teamspace/studios/this_studio/pa4h_v5_run` are preserved.
Helios changes must never be applied to that directory or these Alibaba sources.

## Helios: corrected seed 1 is active

The prior 50% training pilot completed 300 episodes. Its validation at 5% had
zero rack violations for all normal policies, so that window cannot distinguish
power protection. At 50% validation stress, RPA had reward 39.107 versus
MARL sequence 38.911 and pointwise 38.938; all three had zero rack violations.
These are **prior-pilot results**, not results from the corrected actor.
Raw compact tables are under `results/helios_prior_*.csv`.

Largest investigated limitation: the linear rack score added shared job context
to every action equally, cancelling its effect on relative softmax probabilities.
The corrected nonlinear head also attaches the counterfactual utility to its
corresponding action. See `docs/helios_conditioned_head.md` for evidence.
The repair and finite-zero counterfactual loss are in
`reproduction/helios_conditioned.patch`, applied only to an isolated Helios copy.
This architecture requires fresh training, not an old checkpoint resume.

Current server source:
`/teamspace/studios/this_studio/pa4h_helios_conditioned_20260927`.
Tmux: `helios_conditioned_seed1`.
Output: `outputs/helios_conditioned_300_seed1` inside that source directory.
300 episodes; training hours 1000–1010 at 50%; validation hours 1010–1020
at 5% and 50%, both decoders. Holdout hours 1020–1030 remain reserved.
The runner verifies/snapshots before evaluation. No seed 2–5 queue is active.
The earlier queue was stopped; its partial seed 2 is not a completed replication.

Check status without changing the run:

```bash
ssh s_01kr484z6kd47ymw8a35sjcg1w@ssh.lightning.ai \
  'cd /teamspace/studios/this_studio/pa4h_helios_conditioned_20260927; tail -30 outputs/helios_conditioned_300_seed1/live/train.log'
```

After completion, compare corrected seed 1 with RPA and the prior pilot using
these same evaluation windows before launching seeds 2–5. Do not infer an
improvement from training replays alone or change protocols to match the draft.
