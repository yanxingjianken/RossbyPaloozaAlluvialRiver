# Rossby Palooza — Alluvial Rivers & Jet Streams

Group project treating **river meanders via depth-averaged vorticity dynamics**: the
parabolic in-channel jet carries a constant positive vorticity gradient 2Δ/b² — the
channel analogue of the planetary β (hence the name). The working theory (6/30 deck)
predicts, unlike classical bend theory, **upstream-propagating** meanders driven by an
erodible-bank / channel-Rossby-wave cooperation.

## Layout

| path | what |
|---|---|
| [`numerical/`](numerical/) | one **machine-verified explainer package** per literature item + the group's own theory — see [`numerical/README.md`](numerical/README.md) for the five-source map and house rules |
| [`numerical/vorticity_meander/THEORY.md`](numerical/vorticity_meander/THEORY.md) | the governing equations, 3D→2D reduction, and the term-by-term contrast with Ikeda et al. (1981) Eq. (7) |
| [`numerical/dedalus_meander/`](numerical/dedalus_meander/) | Dedalus v3 2D channel model of those equations (EVP + IVP: erodible-bank sin(kx) initial planforms) |
| [`literature/`](literature/) | source PDFs incl. the 6/30 group-meeting deck |

## ⚠ Visibility

This repository is **private** and must stay private as long as `literature/` contains
the four published (copyrighted) paper PDFs. Before any switch to public, strip them
from the **entire git history** (`git filter-repo`), not just the tip.

## Sync

The repo auto-syncs at the end of every Claude Code session via
`scripts/sync_github.sh` (SessionEnd hook; pull --rebase, leak guard, push — aborts on
conflict, never force-pushes). Run the same script manually anytime:

```bash
bash scripts/sync_github.sh
```
