# EVESeek — Design Document v2.0

**Purpose**: Authoritative reference for architecture, algorithms, and planned features.
Detailed enough for a complete rebuild from scratch.

---

## 1. Project Goal

Manufacturing cost calculator for EVE Online. Given an item to build, compute the full
supply-chain cost: raw materials → intermediates → finished product, including all
manufacturing and reaction job fees, refining fees, and logistics. The user picks the
item, system, structure, and market source; the tool returns a fully costed BOM tree
and a human-readable breakdown.

---

## 2. Tech Stack

| Layer | Choice | Reason |
|---|---|---|
| Language | Python 3.11+ | |
| Web framework | FastAPI (async) | Auto docs at /docs, Pydantic validation |
| ASGI server | Uvicorn | `uvicorn app.main:app --reload` |
| Static data | Fuzzwork SQLite `data/eve.db` (528 MB) | Complete SDE, includes volume data missing from Hoboleaks |
| Live data | ESI (CCP public REST API) | Market orders, adjusted prices, system cost indices |
| HTTP client | httpx (sync) | Simple, no async needed for blocking ESI calls |
| Validation | Pydantic v2 | |
| Precision | Python `Decimal` for all ISK | Never float for money |
| Package manager | mamba (conda env at `/opt/homebrew/Caskroom/miniconda/base`) | |
| Frontend | Vanilla HTML/CSS/JS, no build step | Served as FastAPI StaticFiles |

Dependencies (`requirements.txt`, gitignored):
```
fastapi
uvicorn[standard]
httpx
```

---

## 3. Repository Layout

```
EVESeek/
├── app/
│   ├── main.py              # FastAPI app, mounts router and static files
│   ├── api/
│   │   └── routes.py        # All HTTP endpoints
│   ├── engine/
│   │   ├── bom.py           # Recursive BOM engine
│   │   ├── compare.py       # compare-material-source engine
│   │   └── refining.py      # Standalone refining cost calculator
│   ├── esi/
│   │   ├── client.py        # ESI HTTP calls + in-memory TTL cache + disk cache L2
│   │   └── disk_cache.py    # SQLite persistent cache (data/esi_cache.db)
│   ├── data/
│   │   └── sde.py           # All SQLite queries against eve.db
│   ├── models/
│   │   └── schemas.py       # Pydantic request/response models
│   └── core/
│       └── utils.py         # Pure EVE math (no I/O)
├── static/
│   ├── index.html           # Single-page UI
│   ├── style.css            # Dark theme styles
│   └── app.js               # Autocomplete, fetch, BOM tree rendering
├── data/
│   ├── eve.db               # Fuzzwork SDE SQLite, 528 MB, gitignored
│   └── esi_cache.db         # ESI market order disk cache, gitignored
├── scripts/
│   ├── download_fuzzwork_sde.py
│   └── download_sde.py
├── DESIGN_2.0.md            # This file
├── .gitignore               # Excludes: data/, requirements.txt, DESIGN.md, __pycache__
└── README.md
```

---

## 4. Data Sources

### 4.1 Fuzzwork SDE — `data/eve.db`

SQLite database. Downloaded once, gitignored. Key tables used:

| Table | Purpose | Key Columns |
|---|---|---|
| `invTypes` | Item metadata | `typeID`, `typeName`, `groupID`, `portionSize`, `volume` |
| `invVolumes` | Packaged volumes (ships etc.) | `typeID`, `volume` |
| `invGroups` | Group → category mapping | `groupID`, `categoryID`, `groupName` |
| `invCategories` | Category names | `categoryID`, `categoryName` |
| `industryBlueprints` | Max runs per BPC | `typeID` (blueprint), `maxProductionLimit` |
| `industryActivity` | Activity times | `typeID`, `activityID`, `time` |
| `industryActivityProducts` | What a blueprint produces | `typeID` (bp), `activityID`, `productTypeID`, `quantity` |
| `industryActivityMaterials` | Input materials | `typeID` (bp), `activityID`, `materialTypeID`, `quantity` |
| `invTypeMaterials` | Refining outputs | `typeID` (ore/gas), `materialTypeID`, `quantity` |
| `mapSolarSystems` | System name → ID lookup | `solarSystemID`, `solarSystemName`, `security` |

**Volume query**: Always `COALESCE(invVolumes.volume, invTypes.volume)`.
Ships have a packaged volume in `invVolumes` that differs from their assembled volume in `invTypes`.

**portionSize**: Number of ore/gas units per refining batch. For all ore variants in this SDE
(including compressed), portionSize = 100. `invTypeMaterials.quantity` is the theoretical
max output per batch at 100% yield.

**Activity IDs**:
- `1` = Manufacturing
- `11` = Reaction (moon/composite materials)

### 4.2 ESI (EVE Swagger Interface)

Public API, no auth required for the endpoints used.

| Endpoint | Data | Cache TTL |
|---|---|---|
| `GET /markets/{region_id}/orders/?type_id={id}&page={n}` | All buy+sell orders in region (paginated) | 5 min |
| `GET /markets/prices/` | Universe-wide adjusted prices (EIV) | 24 hr |
| `GET /industry/systems/` | All system cost indices (full universe, one call) | 24 hr |

**Pagination**: ESI market orders return up to 1000 per page. Subsequent pages use `?page=N`.
Total page count is in the `X-Pages` response header. The client must fetch all pages.

**Station IDs for specific-station filtering**:
- Jita 4-4 Caldari Navy Assembly Plant: `location_id = 60003760`, region `10000002`
- Amarr VIII (Oris) Emperor Family Academy: `location_id = 60008494`, region `10000043`

ESI market orders include `location_id` field. Filter on this to get station-specific orders.

**Cost index keys** are strings: `"manufacturing"`, `"reaction"`, `"researching_time_efficiency"`, etc.
Reprocessing is NOT in ESI cost indices — it is a flat user-supplied rate.

---

## 5. Implemented: `app/core/utils.py`

Pure math, no I/O. All formulas replicate in-game behavior.

```python
def ceil_qty(base_qty: int, runs: int, me_level: int) -> int:
    # EVE rounding rule: ceil at every BOM node, never at the final sum
    return math.ceil(base_qty * runs * (1 - me_level * 0.01))
```

```python
def broker_fee_rate(broker_relations, faction_standing, corp_standing) -> Decimal:
    # Minimum 0% (can't go negative)
    rate = 0.03 - 0.003 × broker_relations - 0.0003 × faction_standing - 0.0002 × corp_standing
    return max(rate, 0)
```

```python
def job_cost_manufacturing(adjusted_price, system_cost_index, structure_bonus,
                           facility_tax, fw_level) -> Decimal:
    # Used for both manufacturing (activityID=1) and reactions (activityID=11)
    # Reactions: pass facility_tax=0
    SCC_SURCHARGE = 0.04   # fixed 4% governmental levy, always applies
    base = adjusted_price × (system_cost_index × (1 - structure_bonus) + facility_tax + SCC_SURCHARGE)
    return base × (1 - fw_level × 0.1)
```

```python
def job_cost_refining(ore_adjusted_price, reprocessing_rate, structure_bonus,
                      fw_level) -> Decimal:
    # reprocessing_rate is user-supplied flat rate (not from ESI)
    # No SCC surcharge, no facility tax for refining
    base = ore_adjusted_price × reprocessing_rate × (1 - structure_bonus)
    return base × (1 - fw_level × 0.1)
```

---

## 6. Implemented: `app/data/sde.py`

All SQLite queries. Opens a new connection per call (stateless). `sqlite3.Row` factory
allows column access by name. **Important**: `sqlite3.Row` has no `.get()` method — use
direct subscript `row["col"]` only.

```python
get_type(type_id)
# Returns: typeID, typeName, groupID, mass, portionSize,
#          COALESCE(invVolumes.volume, invTypes.volume) AS volume

get_blueprint_for_product(product_type_id)
# Returns the blueprint that produces this item.
# Searches activityID IN (1, 11) — manufacturing and reactions only.
# LIMIT 1: some items have multiple blueprints; first match wins.
# Returns: blueprint_type_id, activityID, quantity (units produced per run)

get_activity_materials(blueprint_type_id, activity_id)
# Returns: materialTypeID, quantity
# These are BASE quantities (before ME reduction).

get_group(group_id)
# Returns: groupID, categoryID, groupName

get_refining_outputs(type_id)
# Returns: materialTypeID, quantity  (from invTypeMaterials)
# quantity = max minerals per portionSize batch at 100% yield

get_activity_time(blueprint_type_id, activity_id)
# Returns base job time in seconds (for future use)

get_max_production_limit(blueprint_type_id)
# Returns maxProductionLimit from industryBlueprints (max runs per BPC copy).
# Returns 1 if blueprint not found.
# Examples: Phoenix Navy Issue Blueprint = 1, Capital Core Temp Reg Blueprint = 40

get_ore_sources_for_mineral(mineral_type_id)
# Returns all compressed ores (categoryID=25, typeName LIKE 'Compressed %') that
# yield this mineral on refining, with portionSize and max_minerals_per_batch.
# Used by compare-material-source engine.

search_types(query, limit=20)
# WHERE typeName LIKE '%query%' AND published=1

search_solar_systems(query, limit=20)
# SELECT solarSystemID, solarSystemName, security
# FROM mapSolarSystems WHERE solarSystemName LIKE '%query%'
# Used for system autocomplete in the frontend.
```

---

## 7. Implemented: `app/esi/client.py` + `app/esi/disk_cache.py`

Three-level cache for ESI market orders:
- **L1**: In-memory dict `dict[str, (expiry, value)]` — cleared on restart
- **L2**: SQLite `data/esi_cache.db` — survives restarts, same TTL as L1 (5 min for market orders)
- **L3**: ESI HTTP fetch

`disk_cache.py` interface:
```python
def get(key: str) -> Optional[Any]
# SELECT value FROM cache WHERE key = ? AND expires > time.time()
# Value is JSON-serialised, deserialized on read.

def put(key: str, value: Any, ttl: int) -> None
# INSERT OR REPLACE INTO cache (key, value, expires) VALUES (?, ?, time.time()+ttl)
```

`client.py` cache lookup order for `get_market_orders`:
1. Check L1 in-memory dict
2. If miss, check L2 SQLite; promote hit to L1
3. If miss, fetch from ESI; store in both L1 and L2

```python
STATION_REGION: dict[int, int] = {
    60003760: 10000002,  # Jita IV-4 → The Forge
    60008494: 10000043,  # Amarr EFA → Domain
}

get_market_orders(region_id, type_id) -> list[dict]
# Fetches ALL buy and sell orders for this type in the region (all pages).
# Reads X-Pages header to iterate pages. Each order has:
#   price, volume_remain, is_buy_order, location_id, ...
# Cache key: f"orders:{region_id}:{type_id}", TTL 5 min

get_station_fill_price(region_id, type_id, station_id, quantity, is_buy_order) -> Optional[Decimal]
# Filters orders by location_id == station_id and is_buy_order match.
# Sorts sell orders ascending by price (buy orders descending).
# Fills greedily up to quantity, returns volume-weighted average ISK.
# Returns None if no orders at that station.
# If station volume < quantity, returns weighted avg over available volume.

get_adjusted_price(type_id) -> Optional[Decimal]
# Universe-wide CCP-calculated adjusted price (for EIV/job cost base)

get_system_cost_index(system_id) -> dict[str, Decimal]
# Thin wrapper over _get_all_cost_indices().get(system_id, {})
# Full universe fetched and cached once per day under key "all_cost_indices".
# Any subsequent system lookup is a dict lookup — no additional ESI call.
```

---

## 8. Implemented: `app/engine/bom.py`

Recursive BOM engine. Entry point: `build_cost(req: BuildCostRequest)`.

### 8.1 Recursion Stop Conditions

A node is a leaf (buy from market, no further recursion) if ANY of:
1. `group_id in _RAW_GROUP_IDS` — moon materials, ice products, compressed ice, gas cloud
2. `_is_raw_category(type_row)` — categoryID in {25 (Asteroid), 18 (Drone)} via group lookup
3. `blueprint is None` — no manufacturing or reaction blueprint exists (catches minerals like
   Tritanium: they have no blueprint, so they stop here automatically)
4. `depth >= 10` — safety limit

**Critical**: Do NOT add categoryID=4 (Material) to `_RAW_CATEGORY_IDS`. Reaction intermediates
like Phenolic Composites also have categoryID=4. Adding it breaks reaction recursion.
Minerals (Tritanium, Pyerite, etc.) have categoryID=4 but stop via `blueprint is None`.

**Raw groups** (`_RAW_GROUP_IDS`):
- `711` — Moon Materials
- `423` — Ice Products
- `426` — Compressed Ice
- `873` — Gas Cloud

**Raw categories** (`_RAW_CATEGORY_IDS`):
- `25` — Asteroid (ore)
- `18` — Drone

### 8.2 Station-Specific Fill Pricing

```python
_STATION_MAP: dict[str, tuple[int, bool]] = {
    "jita_sell":  (60003760, False),
    "jita_buy":   (60003760, True),
    "amarr_sell": (60008494, False),
    "amarr_buy":  (60008494, True),
}
```

`_market_price(type_id, quantity, req)` looks up `(station_id, is_buy)` from `_STATION_MAP`,
derives `region_id` from `esi.STATION_REGION`, then calls `esi.get_station_fill_price()`.
The actual quantity needed is passed so the fill price reflects real available volume.
Broker fee applied on top for buy orders.

### 8.3 EIV Calculation — CRITICAL

The job fee base (EIV = Estimated Item Value) is:

```
EIV = sum(adjusted_price[material] × base_quantity[material])
      for all input materials of this blueprint activity
```

Use the **input materials' adjusted prices**, NOT the output item's adjusted price.
Many items (especially Navy/faction ships) have adjusted_price = 0, which would give
wrong job fees. The EIV of the inputs is always non-zero.

Example: Phoenix Navy Issue adjusted_price ≈ 0. Its inputs are components with non-zero
adjusted prices. Using input EIV gives ~209M ISK manufacturing fees vs ~1.4M ISK wrong result.

### 8.4 ME Application

```python
def ceil_qty(base_qty, runs, me_level):
    return math.ceil(base_qty * runs * (1 - me_level * 0.01))
```

Per-blueprint ME is looked up per node: `me = req.me_overrides.get(type_id, req.me_level)`.
`me_overrides` is a `dict[int, int]` keyed by product `type_id`. If a type_id has no
override, the global `me_level` (default 10) applies.

### 8.5 Job Fees

Manufacturing fee applies to `activityID=1`. Reaction fee applies to `activityID=11`.
Both use `job_cost_manufacturing()`. Reactions pass `facility_tax=0`.

```python
fee = job_cost_manufacturing(eiv, cost_index, structure_bonus, facility_tax, fw_level) * runs
```

The `* runs` multiplier: EVE charges per run. If the user is building 5 ships, fees are 5×.

### 8.6 BPC Run Count

```python
max_runs_per_bpc = sde.get_max_production_limit(bp_type_id)  # from industryBlueprints
qty_per_run = blueprint["quantity"] or 1                      # from industryActivityProducts
bpc_copies_needed = math.ceil(quantity / (max_runs_per_bpc * qty_per_run))
```

Stored on every non-leaf `BOMNode`. Leaf nodes have `bpc_copies_needed=0`, `max_runs_per_bpc=0`.

### 8.7 CostBreakdown Accumulation

`CostBreakdown` fields accumulate up the tree:
- `material_costs`: leaf market prices
- `manufacturing_fees`: manufacturing job fees at every node
- `reaction_fees`: reaction job fees at every node
- `refining_fees`: always 0 in BOM (refining is a separate calculation)
- `logistics_costs`: volume × logistics_rate at every node (leaf and intermediate)

`total_cost` at each node = sum of all breakdown fields.

---

## 9. Implemented: `app/engine/refining.py`

Standalone refining calculator. NOT part of the BOM engine.

Refining is a separate operation: user deposits ore/gas into a refinery structure,
pays a job fee, receives minerals/materials. This is independent of the manufacturing chain.
All three — manufacturing, reaction, refining — happen in the same system (user confirmed).

**Algorithm**:
1. Look up `portionSize` from `invTypes` (100 for standard ores)
2. `batches = quantity // portionSize` (integer division — partial batches not allowed)
3. For each output mineral: `qty = floor(invTypeMaterials.quantity × reprocessing_yield) × batches`
4. Fee per ore unit: `job_cost_refining(ore_adj_price, reprocessing_rate, structure_bonus, fw_level)`
5. Total fee: `fee_per_unit × (batches × portionSize)`

**Why reprocessing_rate is user-supplied**: ESI `/industry/systems/` does not include
a reprocessing cost index. The rate is set per-structure by its owner.

---

## 10. Implemented: `app/engine/compare.py` — `POST /api/v1/compare-material-source`

Compare buying leaf minerals directly vs buying compressed ore and refining.

### 10.1 Request

Same fields as `BuildCostRequest` plus:
- `reprocessing_yield: float = 0.876`
- `reprocessing_rate: Decimal = 0.02`
- `refinery_bonus: float = 0.0` — separate from `structure_bonus` (EC vs refinery)

### 10.2 Response

```json
{
  "direct_buy": {
    "total_isk": ..., "total_m3": ...,
    "items": [{"type_id", "name", "quantity", "unit_price", "total_isk", "volume_m3"}]
  },
  "compressed_ore": {
    "total_isk": ...,
    "effective_isk": ...,
    "total_m3": ...,
    "refined_total_m3": ...,
    "refining_fee": ...,
    "leftover_total_isk": ...,
    "ore_items": [
      {"ore_type_id", "ore_name", "for_mineral_type_id", "for_mineral_name",
       "quantity", "unit_price", "total_isk", "refining_fee",
       "byproduct_credit", "effective_isk", "volume_m3", "refined_m3"}
    ],
    "direct_items": [...],
    "leftover_items": [
      {"type_id", "name", "quantity", "buy_price", "total_isk"}
    ]
  }
}
```

**Field semantics**:
- `total_isk`: raw ore purchase price (what you put up front)
- `effective_isk`: `total_isk + refining_fee − byproduct_credit` (apples-to-apples vs direct buy)
- `total_m3`: compressed ore haul volume (what you move to the refinery)
- `refined_total_m3`: total mineral volume after refining (what leaves the refinery)
- `leftover_total_isk`: Jita buy value of surplus minerals (sell back to recoup cost)
- `leftover_items`: per-mineral surplus that exceeds what the build consumes

`direct_items` in `compressed_ore`: items with no compressed ore source (gas, moon
materials). These are bought directly in both paths so totals remain comparable.

### 10.3 Algorithm

1. Run the BOM engine internally to get leaf node quantities (pass `logistics=0`).
2. Aggregate leaves by `type_id` across the whole tree (`_collect_leaves` recurses).
3. Pre-warm ESI cache: `_prefetch_ore_orders()` fetches all compressed ore type_ids
   and their refining byproducts in parallel using `ThreadPoolExecutor(max_workers=20)`.
   This runs before the sequential per-mineral loop so all subsequent price lookups are
   cache hits.
4. For each leaf, call `get_ore_sources_for_mineral(type_id)` (returns compressed ores only).
5. For each compressed ore candidate:
   ```
   minerals_per_batch = floor(max_minerals_per_batch × reprocessing_yield)
   batches            = ceil(mineral_qty / minerals_per_batch)
   ore_units          = batches × portionSize
   ore_unit_price     = get_station_fill_price(...)    # None → skip this ore
   ```
6. Compute byproduct credit (all refining outputs, including excess of target mineral):
   ```
   for each output of this ore:
       output_qty = floor(output.quantity × yield) × batches
       if output is the target mineral:
           excess = output_qty - mineral_qty
           credit += jita_buy_price[target] × excess   # excess from ceil rounding
       else:
           credit += jita_buy_price[output] × output_qty
   ```
7. `effective_cost = ore_unit_price × ore_units + refining_fee - credit`
8. Pick ore with lowest `effective_cost` for each mineral.
9. Items with no ore source go into `direct_items` in both paths.

### 10.4 True Global Leftover Calculation

The per-mineral credit in step 6 slightly over-counts when the same ore produces two
needed minerals (e.g. Scordite covers both Tritanium and Pyerite). Each mineral's
independent ore selection credits the other's byproduct.

**Fix**: After the per-mineral loop, compute the true global leftover:

```python
total_produced: dict[int, (name, qty)] = {}
for each chosen ore_item:
    batches = ore_item.quantity // portionSize
    for each refining output:
        output_qty = floor(quantity × yield) × batches
        total_produced[out_type_id] += output_qty

leftover_items = []
for type_id, (name, qty_produced) in total_produced.items():
    qty_needed = leaves.get(type_id, 0)
    surplus = qty_produced - qty_needed
    if surplus > 0:
        buy_price = jita_buy_price(type_id, surplus)
        leftover_items.append(LeftoverItem(...))
```

This is the only correct surplus figure. The per-item `byproduct_credit` fields in
`ore_items` remain slightly over-counted and are only used for per-ore comparison, not
the global leftover total.

---

## 11. Implemented: `app/api/routes.py`

```
POST /api/v1/build-cost               → BuildCostRequest → BuildCostResponse
POST /api/v1/refine-cost              → RefineCostRequest → RefineCostResponse
POST /api/v1/compare-material-source  → CompareMaterialSourceRequest → CompareMaterialSourceResponse
GET  /api/v1/search?q=                → [{type_id, name}]
GET  /api/v1/search-systems?q=        → [{system_id, name, security}]
```

`/search-systems` security value is rounded to 1 decimal place in the response.

---

## 12. Implemented: `static/` — Frontend

Vanilla HTML/CSS/JS. No build step. Served via `StaticFiles(directory="static", html=True)`.

Three files: `index.html`, `style.css`, `app.js`.

### UI Features

- **Item search**: debounced autocomplete against `/api/v1/search`. Stores `type_id` in hidden field.
- **System search**: debounced autocomplete against `/api/v1/search-systems`. Shows security
  status colour-coded (green ≥0.5, yellow 0.1–0.5, red <0.1).
- **Calculate**: fires `/build-cost` and `/compare-material-source` in parallel (`Promise.all`).
  Results appear together; if compare fails, build results still show.
- **Cost breakdown table**: shows each component as ISK and % of total. When leftover credit
  is present, switches to "Net total cost" headline and recalculates all percentages against
  the net total (ore path costs replace direct-buy material costs).
- **BPC table**: all non-leaf BOM nodes with `bpc_copies_needed > 0`, collected by
  `collectBPC()` tree walk. Shows copies needed, max runs/copy, total runs.
- **Compare table**: Direct buy vs compressed ore side-by-side. Shows:
  - Net material cost (ore − leftover credit)
  - Ore purchase cost (sub-row)
  - Leftover credit (sub-row)
  - Refining fee
  - Net total
  - Volume to haul (compressed ore m³)
  - Volume after refining (mineral m³)
  Winning path highlighted green.
- **Ore breakdown table**: one row per ore item. Shows ore cost, refining fee, byproduct
  credit, effective ISK, volume m³. `direct_items` (no ore source) shown as "direct buy" rows.
- **Shopping list section**: two tabs, EVE multibuy format (`Item Name x Quantity` per line).
  - Tab 1 "Compressed ore": compressed ore items + `-- Sell leftovers --` separator + leftover items
  - Tab 2 "Direct buy": minerals bought directly
  - Tab labels show upfront purchase cost (ore tab = `co.total_isk`, direct tab = `db.total_isk`)
  - Cheaper tab marked with ✓; cheaper determination uses `co.effective_isk` vs direct cost
  - Defaults to the cheaper tab
  - Copy button: copies textarea content, briefly shows "Copied!" confirmation
- **Leftover materials section**: table of surplus minerals after refining, with Jita buy
  price per unit and total ISK. Total leftover value shown in footer.
- **BOM tree**: recursive DOM rendering. Click any parent node to collapse/expand children.
  Shows quantity, total cost, and BPC info per node.
- **Enter key**: triggers Calculate when focus is on an input (unless an autocomplete
  dropdown is open).

### Cache-busting

`app.js` is served with a `?v=N` query string in `index.html`. Increment `N` on every
change to force browsers to fetch the updated file.

### Dropdowns

- **Buy from**: `jita_sell | jita_buy | amarr_sell | amarr_buy`
- **FW Bonus**: `None (0) | Level 1–5`
- **Structure bonus**: `EC no rig (1%) | EC + T1 rig (4%) | EC + T2 rig (5.5%)`

---

## 13. Structure + Rig Bonus

### Manufacturing Job Cost (`structure_bonus` field)

| Configuration | structure_bonus |
|---|---|
| NPC station | Not modeled |
| Engineering Complex, no rig | 0.01 (1% base passive bonus) |
| EC + T1 capital manufacturing rig | 0.04 (1% + 3%) |
| EC + T2 capital manufacturing rig | 0.055 (1% + 4.5%) |

**Critical finding**: Manufacturing rigs (Standup L-Set/M-Set Manufacturing Efficiency)
provide **material reduction** and **time reduction** only. `Cost Reduction Bonus`
(attribute 2595) is 0.0 for all manufacturing rigs in the SDE. Cost reduction rigs
only exist for lab activities (invention, ME/TE research, copying).

Therefore `structure_bonus` for manufacturing job cost does not depend on rig choice.
The T1/T2 values in the dropdown are the EC passive bonus plus the rig ME bonus expressed
as a job-cost equivalent for convenience — verify these against current patch notes.

### Refinery (`reprocessing_yield` field)

Rigs for Tatara/Athanor affect `reprocessing_yield`, not the fee formula.

| Rig | yield bonus |
|---|---|
| Standup M-Set Ore Reprocessing I | +2% yield |
| Standup M-Set Ore Reprocessing II | +3% yield |

---

## 14. Pydantic Models — `app/models/schemas.py`

### ISK Type (critical)

```python
ISK = Annotated[Decimal, PlainSerializer(lambda v: float(v), return_type=float)]
```

All Decimal response fields use this type. Without it, Pydantic v2 serializes Decimal
as a string in some nested models and as a float at the top level — inconsistent JSON.
`json_encoders` in `model_config` does NOT cascade to nested models in Pydantic v2.

### BuildCostRequest

```python
class BuildCostRequest(BaseModel):
    type_id: int
    system_id: int                  # manufacturing/reaction/refining system (all same)
    me_level: int = Field(10, ge=0, le=10)
    me_overrides: dict[int, int] = Field(default_factory=dict)  # type_id → ME level
    structure_bonus: float = Field(0.01, ge=0.0, le=0.10)
    fw_level: int = Field(0, ge=0, le=5)
    runs: int = Field(1, ge=1)
    material_source: Literal["jita_sell", "jita_buy", "amarr_sell", "amarr_buy"] = "jita_sell"
    facility_tax: Decimal = Field(Decimal("0.0025"), ge=0)
    logistics_cost_isk_per_m3: Decimal = Field(Decimal("0"), ge=0)
    broker_relations_level: int = Field(0, ge=0, le=5)
    faction_standing: float = Field(0.0, ge=-10.0, le=10.0)
    corp_standing: float = Field(0.0, ge=-10.0, le=10.0)
```

`region_id` was removed — derived from `material_source` via `esi.STATION_REGION`.

### BOMNode

```python
class BOMNode(BaseModel):
    type_id: int
    name: str
    quantity: int
    cost_per_unit: ISK
    total_cost: ISK
    cost_breakdown: CostBreakdown
    children: list["BOMNode"] = []
    bpc_copies_needed: int = 0      # 0 for leaf nodes
    max_runs_per_bpc: int = 0       # 0 for leaf nodes
```

`BOMNode.model_rebuild()` must be called after class definition to resolve the
forward reference in `children: list["BOMNode"]`.

### CompressedOreItem / CompressedOrePath / LeftoverItem

```python
class CompressedOreItem(BaseModel):
    ore_type_id: int; ore_name: str
    for_mineral_type_id: int; for_mineral_name: str
    quantity: int; unit_price: ISK
    total_isk: ISK          # raw ore purchase price
    refining_fee: ISK
    byproduct_credit: ISK   # per-ore estimate (slightly over-counted — see section 10.4)
    effective_isk: ISK      # total_isk + refining_fee − byproduct_credit
    volume_m3: ISK          # compressed ore volume (haul to refinery)
    refined_m3: ISK         # total mineral volume produced after refining

class LeftoverItem(BaseModel):
    type_id: int; name: str
    quantity: int; buy_price: ISK; total_isk: ISK

class CompressedOrePath(BaseModel):
    total_isk: ISK          # raw ore purchase price across all ore items
    effective_isk: ISK      # total_isk + refining_fee − all per-ore byproduct credits
    total_m3: ISK           # compressed ore volume + non-mineral direct volume
    refined_total_m3: ISK   # total mineral volume after refining + non-mineral direct
    refining_fee: ISK
    ore_items: list[CompressedOreItem]
    direct_items: list[DirectBuyItem]
    leftover_items: list[LeftoverItem] = []
    leftover_total_isk: ISK = Decimal("0")
```

---

## 15. Known Bugs / Gotchas

1. **EIV must use input materials, not output item**. Many faction/Navy items have
   `adjusted_price = 0` from ESI, giving near-zero job fees. The correct EIV is the
   sum of input materials' adjusted prices × base quantities.

2. **Mineral categoryID is 4, not 22**. Minerals (Tritanium etc.) are in category 4
   (Material). They have no blueprint, so they stop recursion via `blueprint is None`.
   Do not add categoryID=4 to `_RAW_CATEGORY_IDS` — it also matches reaction
   intermediates (Phenolic Composites, Isogen-10, etc.) and will break reaction recursion.

3. **ESI cost index keys are strings**. `cost_indices.get("manufacturing")`, not `get(1)`.

4. **`invVolumes` vs `invTypes.volume`**. Ships have a packaged volume (e.g., 500,000 m³
   assembled vs 50,000 m³ packaged). Always use `COALESCE(invVolumes.volume, invTypes.volume)`.

5. **Reactions: no facility tax**. The `job_cost_manufacturing` function is reused for
   reactions by passing `facility_tax=Decimal("0")`.

6. **portionSize in this SDE**: Compressed ore variants (e.g., Compressed Veldspar)
   show portionSize=100 — same as regular ore. Both have the same mineral yield per 100
   units in invTypeMaterials. The difference between compressed and regular ore is
   purely volume (m³ per unit), not mineral yield per portionSize batch.

7. **Manufacturing rigs do not reduce job cost**. Standup L-Set/M-Set manufacturing rigs
   only affect material efficiency (attribute 2594) and time efficiency (attribute 2593).
   `Cost Reduction Bonus` (attribute 2595) is 0.0 on all manufacturing rigs in the SDE.

8. **Compare: ore sourced per-mineral independently**. The compare engine picks the best
   ore for each mineral separately. It does not solve the global optimisation problem
   (buying Scordite to cover both Tritanium and Pyerite simultaneously would be more
   efficient). The global leftover calculation (section 10.4) is correct; only the
   per-ore `byproduct_credit` fields are slightly over-counted. A globally optimal
   solution would require LP.

9. **`sqlite3.Row` has no `.get()` method**. Always use direct subscript `row["col"]`.
   `.get()` silently fails on Row objects and causes an AttributeError.

10. **Cost path consistency**: When leftover credit is present, the frontend must use the
    ore path costs (`co.total_isk + co.refining_fee`) as the material cost basis in the
    breakdown — not the direct-buy `material_costs` field. Mixing the two paths gives a
    negative net total.

---

## 16. Running Locally

```bash
# Install deps (mamba/conda env)
mamba install fastapi uvicorn httpx

# Download SDE (one-time, ~528 MB)
python scripts/download_fuzzwork_sde.py

# Run server
uvicorn app.main:app --reload --port 8005

# Open UI
open http://localhost:8005

# API docs
open http://localhost:8005/docs

# Test build cost (Phoenix Navy Issue, Turnur system, ME10, Jita sell)
curl -X POST http://localhost:8005/api/v1/build-cost \
  -H "Content-Type: application/json" \
  -d '{"type_id":73793,"system_id":30002086,"me_level":10,"material_source":"jita_sell"}'

# Test compare (same item)
curl -X POST http://localhost:8005/api/v1/compare-material-source \
  -H "Content-Type: application/json" \
  -d '{"type_id":73793,"system_id":30002086,"me_level":10,"material_source":"jita_sell"}'

# Test refining (10000 Veldspar, 87.6% yield, 2% fee)
curl -X POST http://localhost:8005/api/v1/refine-cost \
  -H "Content-Type: application/json" \
  -d '{"type_id":1230,"quantity":10000,"reprocessing_yield":0.876,"reprocessing_rate":0.02}'

# Test system search
curl "http://localhost:8005/api/v1/search-systems?q=Jita"
```

Expected output for Phoenix Navy Issue (prices fluctuate with live market):
- Total cost: ~2.33B ISK
- Manufacturing fees: ~209M ISK
- Reaction fees: ~5M ISK
- `bpc_copies_needed: 1`, `max_runs_per_bpc: 1` on the root node
