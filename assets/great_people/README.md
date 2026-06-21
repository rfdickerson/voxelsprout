# Great People portraits

The portrait atlas **`great_people.png`** is a **4 × 3 grid** of **362 px** cells
(1448 × 1086 total). Cell `[col, row]` → pixel rect `[col*362, row*362, 362, 362]`.

## Art ↔ code contract

The single source of truth is the gameplay catalog
[`mods/base/data/great_people.json`](../../mods/base/data/great_people.json). Each
figure's `portraitCell: [col, row]` indexes a cell, and the app registers the
figure's `id` as the icon name (so a panel's `portraitName == id` resolves). The app
computes the atlas metadata from that catalog at startup
(`App::setupDemoUi`, the "Great People portrait atlas" block) — there is no separate
layout JSON to keep in sync.

`great_people.json` in this folder is the original art manifest from the asset drop
and is kept for reference only; the loaded gameplay catalog lives under `mods/base`.

| Cell | Figure | Class |
|---|---|---|
| `[0,0]` | Euclid | Great Scientist |
| `[1,0]` | Homer | Great Writer |
| `[2,0]` | Archimedes | Great Engineer |
| `[3,0]` | Aristotle | Great Scientist |
| `[0,1]` | Hypatia | Great Scientist |
| `[1,1]` | Sappho | Great Writer |
| `[2,1]` | Sun Tzu | Great General |
| `[3,1]` | Confucius | Great Philosopher |
| `[0,2]` | Herodotus | Great Writer |
| `[1,2]` | Pythagoras | Great Scientist |
| `[2,2]` | Zhang Heng | Great Engineer |
| `[3,2]` | Aesop | Great Writer |

If the atlas is missing, the Great People window falls back to neutral placeholder
swatches, so the feature stays functional without art.
