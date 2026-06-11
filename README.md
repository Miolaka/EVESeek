# EVESeek

EVE Online manufacturing cost calculator. Given an item to build, computes the full
supply-chain cost from raw materials to finished product — replicating in-game formulas
exactly for job fees, material efficiency, and market fill prices.

## Features

- **Full recursive BOM** with manufacturing + reaction job fees, ME bonus, FW bonus
- **T1 hull build/buy toggle** — when building a T2 ship, manufacture the T1 hull (default) or price it off market; default is build because buying a packaged hull is impractical to haul
- **Station-specific fill pricing** — Jita 4-4 or Amarr EFA, sell or buy side, volume-weighted
- **Compressed ore comparison** — side-by-side direct buy vs buy compressed ore + refine, with logistics-adjusted byproduct credit and true global leftover surplus
- **Leftover optimisation** — set a max leftover value (ISK) and/or a haul cost (ISK/m³); the engine greedily swaps ore choices to stay within the limit; falls back to direct buy with a note if the constraint is impossible
- **Shopping list** — EVE multibuy format, two tabs (compressed ore / direct buy), cheapest tab pre-selected
- **Leftover materials** — surplus minerals with gross value, haul cost, and net credit columns
- **ESI disk cache** — SQLite persistent cache survives server restarts; parallel pre-fetch before compare loop
- **BPC copy counts** at every BOM node

## Stack

- **Backend**: Python 3.11+, FastAPI, Uvicorn
- **Static data**: Fuzzwork SDE SQLite (`data/eve.db`, 528 MB) — blueprints, materials, volumes, systems
- **Live data**: ESI public API — market orders, adjusted prices, system cost indices
- **HTTP client**: httpx (sync)
- **Precision**: Python `Decimal` for all ISK values
- **Frontend**: Vanilla HTML/CSS/JS, no build step

## Setup

**1. Install dependencies**
```bash
mamba install fastapi uvicorn httpx -c conda-forge
```

**2. Download SDE** (~528 MB, one-time)
```bash
python scripts/download_fuzzwork_sde.py
```

**3. Run**
```bash
uvicorn app.main:app --reload --port 8005
```

Open `http://localhost:8005` — API docs at `http://localhost:8005/docs`.

## API

### `POST /api/v1/build-cost`

```json
{
  "type_id": 73793,
  "system_id": 30002086,
  "runs": 1,
  "me_level": 10,
  "material_source": "jita_sell",
  "structure_bonus": 0.01,
  "fw_level": 0,
  "facility_tax": 0.0025,
  "logistics_cost_isk_per_m3": 0,
  "build_t1_hull": true
}
```

`material_source`: `"jita_sell"` | `"jita_buy"` | `"amarr_sell"` | `"amarr_buy"`

`build_t1_hull`: when building a T2 ship, `true` (default) manufactures the T1 hull
from minerals; `false` prices it off the market instead.

Returns: `total_cost`, `cost_breakdown` (material / manufacturing / reaction / logistics),
and a full `bom_tree` with `bpc_copies_needed` and `max_runs_per_bpc` at each node.

### `POST /api/v1/compare-material-source`

Same fields as build-cost, plus:

```json
{
  "reprocessing_yield": 0.876,
  "reprocessing_rate": 0.02,
  "refinery_bonus": 0.0,
  "leftover_logistics_isk_per_m3": 500,
  "max_leftover_isk": 50000000
}
```

`leftover_logistics_isk_per_m3`: ISK/m³ cost to haul surplus minerals to Jita. Deducted
from byproduct credit during ore selection — ores with bulky byproducts score worse when
this is non-zero.

`max_leftover_isk`: optional cap on net leftover value. The engine runs a greedy swap loop
(up to 30 iterations) to find ore combinations that satisfy the limit. If impossible,
`leftover_constraint_met: false` is returned and the frontend falls back to direct-buy price.

Returns: `direct_buy` and `compressed_ore` paths with ISK totals, m³ volumes, per-ore
breakdown, and leftover surplus with `volume_m3`, `logistics_isk`, and `net_isk` per item.

### `POST /api/v1/refine-cost`

```json
{
  "type_id": 1230,
  "quantity": 10000,
  "reprocessing_yield": 0.876,
  "reprocessing_rate": 0.02,
  "structure_bonus": 0.0,
  "fw_level": 0
}
```

Returns: refining fee and list of mineral outputs with quantities.

### `GET /api/v1/search?q=phoenix`

Returns `[{type_id, name}]` for published items matching the query.

### `GET /api/v1/search-systems?q=jita`

Returns `[{system_id, name, security}]` for matching solar systems.
