# EVESeek — Archive v1.0

**Date**: 2026-06-10
**Git ref**: e059fa4
**Status**: Feature-complete v1 — all planned features shipped and pushed to GitHub

---

## What Was Built

EVESeek is an EVE Online manufacturing cost calculator. The user inputs an item to build,
a system, structure settings, and market source. The tool returns the full supply-chain
cost: raw materials → intermediates → finished product, including every in-game fee,
material efficiency bonus, and logistics cost.

---

## Development Phases

### Phase 1 — Foundation
- Replaced Flask prototype with FastAPI scaffold
- Set up Fuzzwork SDE SQLite (`data/eve.db`, 528 MB) as the static data source; Hoboleaks
  JSON lacked volume data so Fuzzwork was chosen instead
- ESI client with in-memory TTL cache (market orders 5 min, adjusted prices 24 hr,
  cost indices 24 hr)
- Pure EVE math utilities: `ceil_qty`, `broker_fee_rate`, `job_cost_manufacturing`,
  `job_cost_refining`

### Phase 2 — BOM Engine (`app/engine/bom.py`)
- Recursive BOM with manufacturing (activityID=1) and reaction (activityID=11) support
- Station-specific volume-weighted fill pricing (Jita 4-4, Amarr EFA; buy and sell sides)
- Per-blueprint ME override (`me_overrides: dict[int, int]`)
- EIV (job fee base) computed from input materials' adjusted prices — NOT the output
  item's adjusted price (faction/Navy ships have adjusted_price ≈ 0)
- BPC copy counting at every non-leaf node using `industryBlueprints.maxProductionLimit`
- Logistics cost: volume × ISK/m³ at every BOM node
- Recursion stop conditions: raw groups (moon materials, ice, gas), raw categories
  (asteroid, drone), no blueprint, depth ≥ 10

### Phase 3 — Refining (`app/engine/refining.py`)
- Standalone refining cost calculator
- `batches = quantity // portionSize` (integer division — partial batches disallowed)
- Output qty: `floor(invTypeMaterials.quantity × reprocessing_yield) × batches`
- Fee: `job_cost_refining(ore_adj_price, reprocessing_rate, refinery_bonus, fw_level)`

### Phase 4 — Compare Engine (`app/engine/compare.py`)
- `POST /api/v1/compare-material-source`: direct buy vs compressed ore + refine
- Per-mineral ore selection: picks compressed ore with lowest effective cost
  (`ore_cost + refining_fee − byproduct_credit`)
- Byproduct credit priced at Jita buy orders regardless of user's material_source
- ESI disk cache (L2 SQLite `data/esi_cache.db`) added so cache survives restarts
- Parallel pre-fetch: `ThreadPoolExecutor(max_workers=20)` warms all ore + byproduct
  market orders before the sequential per-mineral loop — reduced 1-paladin calculation
  from minutes to seconds
- True global leftover: after the per-mineral loop, sums ALL refining outputs from ALL
  chosen ores, subtracts what the build needs, and only the surplus is leftover. This
  fixes the double-counting that occurs when the same ore covers two needed minerals
- Leftover items priced at Jita buy; exposed as `leftover_items` + `leftover_total_isk`
- Volume fields: `total_m3` = compressed ore haul volume; `refined_total_m3` = mineral
  volume after refining

### Phase 5 — Frontend (`static/`)
- Vanilla HTML/CSS/JS, no build step, dark EVE-inspired theme
- Item and system autocomplete with debouncing
- Build-cost and compare fired in parallel (`Promise.all`); results appear together
- Cost breakdown table with ISK and % columns; zero rows hidden
- Net total headline: when leftover credit exists, switches to "Net total cost" and
  uses ore path costs (`co.total_isk + co.refining_fee`) as the material cost basis —
  prevents mixing direct-buy and ore-path numbers
- Compare table: net material cost (ore − leftover), sub-rows for ore purchase and
  leftover credit, refining fee, net total, volume to haul, volume after refining
- Shopping list section: two tabs in EVE multibuy format (`Item Name x Quantity`)
  - Compressed ore tab: ore items + `-- Sell leftovers --` separator + leftover items
  - Direct buy tab: minerals bought directly
  - Tab labels show upfront purchase cost; cheaper tab marked ✓ (comparison uses
    effective cost, not purchase cost)
  - Copy button with "Copied!" flash
- Leftover materials section: surplus minerals, Jita buy price, total value
- BPC table: copies needed, max runs/copy, total runs at every non-leaf node
- Collapsible BOM tree: click any node to expand/collapse

---

## Key Formulas

### Material quantity (EVE rounding)
```
qty = ceil(base_qty × runs × (1 − me_level × 0.01))
```
Applied at every BOM node independently — never at the final sum.

### Manufacturing / reaction job fee
```
EIV = sum(adjusted_price[input] × base_qty[input])
fee = EIV × (cost_index × (1 − structure_bonus) + facility_tax + 0.04) × (1 − fw_level × 0.1) × runs
```
Reactions: `facility_tax = 0`.

### Refining job fee (per ore unit)
```
fee = ore_adjusted_price × reprocessing_rate × (1 − refinery_bonus) × (1 − fw_level × 0.1)
```

### Broker fee rate
```
rate = max(0, 0.03 − 0.003×BrokerRelations − 0.0003×factionStanding − 0.0002×corpStanding)
```

### Ore comparison (per mineral)
```
minerals_per_batch = floor(max_minerals_per_batch × reprocessing_yield)
batches            = ceil(mineral_qty / minerals_per_batch)
ore_units          = batches × portionSize
effective_cost     = (ore_unit_price × ore_units) + refining_fee − byproduct_credit
```
Pick ore with lowest `effective_cost`.

### True global leftover
```
for each chosen ore:
    total_produced[mineral] += floor(output.quantity × yield) × batches
surplus[mineral] = total_produced[mineral] − leaves[mineral]
leftover value   = sum(jita_buy_price × surplus) for surplus > 0
```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/api/v1/build-cost` | Recursive BOM with all fees |
| POST | `/api/v1/refine-cost` | Standalone refining cost |
| POST | `/api/v1/compare-material-source` | Direct buy vs compressed ore comparison |
| GET | `/api/v1/search?q=` | Item name autocomplete |
| GET | `/api/v1/search-systems?q=` | Solar system autocomplete |

---

## Critical Bugs Found and Fixed

| Bug | Root Cause | Fix |
|---|---|---|
| `sqlite3.Row` AttributeError | `row.get("col")` doesn't exist on Row objects | Use `row["col"]` direct subscript everywhere |
| Leftover double-counting | Per-mineral byproduct credit counted each ore's full output independently; same byproduct credited 6–8× for a 8-mineral build | Global `total_produced` accumulation post-loop; only true surplus is leftover |
| Negative net total (−2.3B) | Subtracting `leftover_total_isk` from `bd.material_costs` (direct-buy) while ore path was active | When `useOre`, switch all cost breakdown rows to ore path costs (`co.total_isk + co.refining_fee`) |
| Shopping list tab showed wrong cost | Tab displayed `co.effective_isk` (net after credits, ~176M) instead of actual purchase price | Tab label uses `co.total_isk`; cheaper badge comparison uses `co.effective_isk` |
| Browser serving stale JS | No cache-busting on `app.js` | Added `?v=N` query string; increment on each change |
| Slow compare (minutes) | Sequential ESI calls for 100–200 unique type_ids | SQLite disk cache + `ThreadPoolExecutor(20)` parallel pre-fetch |

---

## Architecture Decisions

| Decision | Choice | Reason |
|---|---|---|
| Static data source | Fuzzwork SQLite | Hoboleaks JSON missing volume data needed for logistics |
| ESI cache layers | L1 in-memory + L2 SQLite | Survives restarts; avoids re-fetching on every server reload during development |
| Ore selection strategy | Greedy per-mineral, independent | Simple and fast; global LP optimisation not needed for practical accuracy |
| ISK serialisation | `ISK = Annotated[Decimal, PlainSerializer(float)]` | Pydantic v2 `json_encoders` does not cascade to nested models |
| Leftover accounting | Global post-loop, not per-mineral | Only correct approach; per-mineral credits double-count shared byproducts |
| Frontend framework | Vanilla JS | No build step, no dependencies, fast iteration |

---

## Files

```
app/
  main.py                 FastAPI app entry point
  api/routes.py           All HTTP endpoints
  engine/
    bom.py                Recursive BOM engine
    compare.py            Material source comparison engine
    refining.py           Standalone refining calculator
  esi/
    client.py             ESI HTTP + L1/L2 cache
    disk_cache.py         SQLite persistent cache (data/esi_cache.db)
  data/sde.py             All SQLite queries against eve.db
  models/schemas.py       Pydantic v2 request/response models
  core/utils.py           Pure EVE math (no I/O)
static/
  index.html              Single-page UI
  style.css               Dark theme
  app.js                  Autocomplete, fetch, rendering
scripts/
  download_fuzzwork_sde.py  One-time SDE download
data/
  eve.db                  Fuzzwork SDE, 528 MB (gitignored)
  esi_cache.db            ESI market order cache (gitignored)
```

---

## Not Built (Future Work)

- Global ore optimisation (LP solver to minimise cost when one ore covers multiple minerals)
- Sell-order listing / profit margin calculator
- Multiple structure support (separate EC and refinery system inputs)
- User accounts / saved builds
- Patch-triggered SDE auto-refresh
