# EVESeek

EVE Online manufacturing cost calculator. Given an item to build, computes the full
supply-chain cost from raw materials to finished product — replicating in-game formulas
exactly for job fees, material efficiency, and market fill prices.

## Features

- **Full recursive BOM** with manufacturing + reaction job fees, ME bonus, FW bonus
- **Station-specific fill pricing** — Jita 4-4 or Amarr EFA, sell or buy side, volume-weighted
- **Compressed ore comparison** — side-by-side direct buy vs buy compressed ore + refine, with byproduct credit and leftover surplus valuation
- **Shopping list** — EVE multibuy format, two tabs (compressed ore / direct buy), cheapest tab pre-selected
- **Leftover materials** — surplus minerals from ceil() rounding, priced at Jita buy
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
  "logistics_cost_isk_per_m3": 0
}
```

`material_source`: `"jita_sell"` | `"jita_buy"` | `"amarr_sell"` | `"amarr_buy"`

Returns: `total_cost`, `cost_breakdown` (material / manufacturing / reaction / logistics),
and a full `bom_tree` with `bpc_copies_needed` and `max_runs_per_bpc` at each node.

### `POST /api/v1/compare-material-source`

Same fields as build-cost, plus:

```json
{
  "reprocessing_yield": 0.876,
  "reprocessing_rate": 0.02,
  "refinery_bonus": 0.0
}
```

Returns: `direct_buy` and `compressed_ore` paths with ISK totals, m³ volumes,
per-ore breakdown, and leftover mineral surplus priced at Jita buy.

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
