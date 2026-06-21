# Advisor portraits

Drop a single portrait atlas here named **`advisors.png`**. The layout is fixed by
[`advisors.json`](advisors.json) in this folder.

## Atlas format

- **`advisors.png`** — RGBA PNG, a **2 × 2 grid** of **256 px** cells → **512 × 512** total.
- Cell `[col, row]` → pixel rect `[col*256, row*256, 256, 256]`.

| Cell `[col,row]` | Portrait name | Advisor |
|---|---|---|
| `[0, 0]` | `hlaalu`   | Councilor Dolvas Andrano — Hlaalu Trade Councilor |
| `[1, 0]` | `redoran`  | Warlord Brara Morvayn — Redoran War-Councilor |
| `[0, 1]` | `telvanni` | Magister Therana Sethan — Telvanni Magister |
| `[1, 1]` | `temple`   | Almoner Tholer Dalveni — Tribunal Temple Almoner |

The portrait names must match the `portraitName` values in
`advisorCatalog()` (`src/game/advisor.cc`) — that's the only contract between art
and code.

Until `advisors.png` is added, the Council screen falls back to a placeholder /
leader portrait, so the feature is fully functional without art. To add a 5th
advisor later, switch to a 3 × 2 grid (768 × 512) and add its `icons` entry —
no code change needed beyond a new catalog row.
