# Replay / relabel / height-Q plan (split by size)

The original plan file is 190 KiB, above this repository's 100 KiB
per-file commit limit, so it is stored in four sequential parts. The
content is unmodified apart from link targets that moved when the plan
was brought into the product docs.

| part | sections |
|---|---|
| [part-1-foundation](part-1-foundation.md) | 1-890 |
| [part-2-exec-and-variance](part-2-exec-and-variance.md) | 891-1502 |
| [part-3-persistent-intervention](part-3-persistent-intervention.md) | 1503-2039 |
| [part-4-extra-tests](part-4-extra-tests.md) | 2040-2709 |
## Figures

Two distribution plots belonged to §6/§7 of the plan. They are 156 KiB and
264 KiB, above the 100 KiB commit limit, so they remain in the workspace at
`.opencode/workspace/plans/assets/` (`height-distribution-20261110.png`,
`phase-height-20261110.png`) and are not part of this repository.

## Why the plan lives here

The replay/relabel/FQE work is BRG product work. Its plan, evidence tables and
journal entries now live in the product docs (`docs/plans/`, `docs/journal.md`)
so they version with the code they describe.
