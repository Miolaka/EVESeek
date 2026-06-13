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
def ceil_qty(base_qty: int, runs: int, me_level: int, structure_bonus: float = 0.0) -> int:
    # EVE batch formula: ceil is applied to the TOTAL batch, not per run.
    # max(runs, ...) ensures at least 1 unit per run.
    return max(runs, math.ceil(base_qty * runs * (1 - me_level * 0.01) * (1 - structure_bonus)))
```

**Key invariant**: ceiling is applied once across all runs, not once per run. For small
fractional quantities (e.g. Coolant base=9, ME=10: 9×0.9=8.1), the per-run formula would
give ceil(8.1)×R = 9R, whereas the correct batch formula gives ceil(8.1×R) which is smaller
for any R > 1.

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

All SQLite queries. Uses a **thread-local persistent connection** — one connection per
thread, reused across calls (not opened/closed per query). `sqlite3.Row` factory allows
column access by name. **Important**: `sqlite3.Row` has no `.get()` method — use direct
subscript `row["col"]` only.

Hot lookup functions are decorated with `@lru_cache` (process-level, unbounded within a
session): `get_type`, `get_blueprint_for_product`, `get_activity_materials`, `get_group`,
`get_max_production_limit`, `get_meta_group`. SDE data is read-only and never changes
while the server is running, so caching is always safe.

```python
get_type(type_id)
# Returns: typeID, typeName, groupID, mass, portionSize,
#          COALESCE(invVolumes.volume, invTypes.volume) AS volume

get_blueprint_for_product(product_type_id)
# Returns the blueprint that produces this item.
# Searches activityID IN (1, 11) — manufacturing and reactions only.
# Filters bt.published = 1 to exclude "Test Reaction Blueprint" stubs (unpublished,
# qty=20/run) that exist alongside the real published Reaction Formula (qty=10,000/run).
# ORDER BY p.quantity DESC ensures the highest-output formula wins when duplicates exist.
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

has_copy_activity(blueprint_type_id)
# Returns True only if the blueprint has activityID=5 AND its invTypes row has a
# non-null marketGroupID (i.e. it is purchasable from an NPC market as a BPO).
# Critical: the SDE gives Triglavian ships (Zirnitra, Leshak, Drekavac) and SoCT
# ships (Gnosis, Praxis) activityID=5 in industryActivity, but their blueprint
# type has marketGroupID=NULL — no BPO exists in-game. Without the marketGroupID
# guard these would be classified as copyable and given a fake calculated fee.

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

def evict_expired() -> None
# DELETE FROM cache WHERE expires <= time.time()
# Called once at server startup (via FastAPI lifespan) to prevent unbounded DB growth.
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
1. `group_id in _RAW_GROUP_IDS` — moon materials, ice products, harvestable gas
2. `_is_raw_category(type_row)` — categoryID in {25 (Asteroid), 18 (Drone)} via group lookup
3. `depth > 0 and not req.build_t1_hull and _is_ship(type_row)` — ship-as-ingredient treated
   as market buy when the user opts not to manufacture the T1 hull (see section 8.8)
4. `blueprint is None` — no manufacturing or reaction blueprint exists (catches minerals like
   Tritanium: they have no blueprint, so they stop here automatically)
5. `depth >= 10` — safety limit

**Critical**: Do NOT add categoryID=4 (Material) to `_RAW_CATEGORY_IDS`. Reaction intermediates
like Phenolic Composites also have categoryID=4. Adding it breaks reaction recursion.
Minerals (Tritanium, Pyerite, etc.) have categoryID=4 but stop via `blueprint is None`.

**Raw groups** (`_RAW_GROUP_IDS`):
- `427` — Moon Materials (Atmospheric Gases, Evaporite Deposits, Tungsten, Platinum, etc.)
- `423` — Ice Products (Heavy Water, Liquid Ozone, isotopes, Strontium Clathrates)
- `711` — Harvestable Cloud (gas cloud harvesting: Fullerites, etc.)

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
def ceil_qty(base_qty, runs, me_level, structure_bonus=0.0):
    # EVE batch formula: ceiling applied once across the full batch, not per run.
    return max(runs, math.ceil(base_qty * runs * (1 - me_level * 0.01) * (1 - structure_bonus)))
```

EVE applies the ceiling once to the **entire batch** (base × runs × ME factor × SB factor),
not once per run. For example, Coolant (base=9, ME10, 3 runs): per-run formula gives
`ceil(8.1) × 3 = 27`; batch formula gives `ceil(8.1 × 3) = ceil(24.3) = 25`. The batch
formula is correct. `max(runs, ...)` ensures at least 1 unit per run regardless of ME.
`structure_bonus` (material reduction from EC rigs) is applied multiplicatively inside the
same `ceil()`, not after.

**ME level rules per node** (applied in both `_build_node` and `_compute_flat_bom`):

```python
if activity_id == ACTIVITY_REACTION:
    default_me = 0          # reactions cannot be researched
elif depth == 0 (or type_id == root_type_id):
    default_me = req.me_level   # root item uses user-specified ME
else:
    default_me = 10         # all sub-manufactured items default to ME 10
me = req.me_overrides.get(type_id, default_me)
```

`me_overrides` is a `dict[int, int]` keyed by product `type_id` — editable per-item in
the BPO Research Levels UI table. The frontend resets overrides when a different item is
selected. Reactions and the root hull are read-only in the UI.

`build_cost()` also returns `bpo_list: list[BPOInfo]` — one entry per manufactured/reacted
node in the BOM (discovered by BFS). Used for both the BPO Research Levels table and the
Blueprint Copies section. Fields:

```python
class BPOInfo(BaseModel):
    type_id: int
    name: str
    activity_id: int        # 1 = manufacturing, 11 = reaction
    me_level: int
    is_root: bool = False
    total_runs: int = 0     # total job runs needed across the build
    runs_per_copy: int = 0  # maxProductionLimit from industryBlueprints
    copies_needed: int = 0  # ceil(total_runs / runs_per_copy)
    is_copyable: bool = True  # False for faction/navy/officer → user input in UI
    copy_cost: ISK = Decimal("0")  # in-game copy job fee for all copies combined
```

`is_copyable` is `False` when:
- `activity_id == 11` (reaction — no copying activity)
- `sde.has_copy_activity(bp_type_id)` returns False (no activityID=5, or blueprint has no NPC marketGroupID — covers Triglavian/SoCT drop-only BPCs)
- `meta_group` in `{3, 4, 5, 6, 15}` (Storyline, Faction, Officer, Deadspace, Abyssal)

Copy job fee formula (same as manufacturing, using "copying" cost index):
```
EIV = sum(adjusted_price[material] × base_qty[material]) × copies_needed
copy_cost = EIV × (copying_ci × (1-sb) + facility_tax + 0.04) × (1 - fw_level × 0.1)
```
The "copying" cost index is fetched from `get_system_cost_index()` alongside manufacturing
and reaction indices — all six activity types are returned by one ESI call.

### 8.5 Job Fees

Manufacturing fee applies to `activityID=1`. Reaction fee applies to `activityID=11`.
Both use `job_cost_manufacturing()`. Reactions pass `facility_tax=0`.

```python
runs_needed = math.ceil(quantity / qty_per_run)   # runs for THIS sub-job
eiv = sum(adj_price[m] * base_qty[m] for m in materials) * runs_needed
fee = job_cost_manufacturing(eiv, cost_index, structure_bonus, facility_tax, fw_level)
```

`runs_needed` is computed per node from `quantity / qty_per_run` — it is NOT the
top-level `req.runs`. EIV is pre-scaled by `runs_needed`, so `job_cost_manufacturing`
is called once and returns the total fee for all runs of this sub-job.

### 8.6 BPC Run Count

```python
qty_per_run  = blueprint["quantity"] or 1            # from industryActivityProducts
runs_needed  = math.ceil(quantity / qty_per_run)     # runs needed for this node
max_runs_per_bpc = sde.get_max_production_limit(bp_type_id)
bpc_copies_needed = math.ceil(runs_needed / max_runs_per_bpc)
```

Stored on every non-leaf `BOMNode`. Leaf nodes have `bpc_copies_needed=0`, `max_runs_per_bpc=0`.

### 8.7 T1 Hull Build / Buy Toggle

T2 ship blueprints list the matching T1 hull as a direct input (e.g. Paladin requires
Apocalypse × 1). By default the engine recursively manufactures that hull — same as
every other intermediate. When `req.build_t1_hull = False`, any ship ingredient
(`_SHIP_CATEGORY_ID = 6`) encountered at depth > 0 is treated as a leaf and priced
off market instead.

Default is `True` (build). This matches game practice: minerals weigh nothing packaged,
but buying a completed hull takes 500,000 m³ assembled — impossible to haul economically.

```python
_SHIP_CATEGORY_ID = 6   # EVE category 6 = Ships

def _is_ship(type_row) -> bool:
    group_row = sde.get_group(type_row["groupID"])
    return group_row is not None and group_row["categoryID"] == _SHIP_CATEGORY_ID

# Inside is_raw check:
or (type_row and type_row["groupID"] and depth > 0
    and not req.build_t1_hull and _is_ship(type_row))
```

T1 hull ingredients are always minerals + components, never another ship, so setting
`build_t1_hull = True` will never trigger the ship-leaf check deeper in the tree.

### 8.9 CostBreakdown Accumulation

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
- `leftover_logistics_isk_per_m3: Decimal = 0` — ISK/m³ cost to haul surplus minerals to Jita
- `max_leftover_isk: Decimal | None = None` — upper bound on net leftover value; `None` = no limit

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
    "leftover_logistics_isk": ...,
    "leftover_net_isk": ...,
    "leftover_constraint_met": true,
    "ore_items": [
      {"ore_type_id", "ore_name", "for_mineral_type_id", "for_mineral_name",
       "quantity", "unit_price", "total_isk", "refining_fee",
       "byproduct_credit", "effective_isk", "volume_m3", "refined_m3"}
    ],
    "direct_items": [...],
    "leftover_items": [
      {"type_id", "name", "quantity", "buy_price", "total_isk",
       "volume_m3", "logistics_isk", "net_isk"}
    ]
  }
}
```

**Field semantics**:
- `total_isk`: raw ore purchase price (what you put up front)
- `effective_isk`: sum of per-ore `(total_isk + refining_fee − byproduct_credit)` — used
  for ore selection only; byproduct credits are logistics-adjusted (see 10.3 step 6)
- `total_m3`: compressed ore haul volume (what you move to the refinery)
- `refined_total_m3`: total mineral volume after refining (what leaves the refinery)
- `leftover_total_isk`: gross Jita buy value of all surplus minerals
- `leftover_logistics_isk`: total cost to haul surplus minerals to Jita
- `leftover_net_isk`: `leftover_total_isk − leftover_logistics_isk` — actual cash received
- `leftover_constraint_met`: `false` when `max_leftover_isk` was set but could not be satisfied
- `LeftoverItem.volume_m3`: total m³ of this surplus item
- `LeftoverItem.logistics_isk`: haul cost for this item = `volume_m3 × rate`
- `LeftoverItem.net_isk`: `total_isk − logistics_isk` for this item (≥ 0)

`direct_items` in `compressed_ore`: items with no compressed ore source (gas, moon
materials). These are bought directly in both paths so totals remain comparable.

### 10.3 Algorithm — `_ore_candidates_for_mineral`

For each mineral, collect ALL valid ore candidates (not just the cheapest):

1. Run the BOM engine internally to get leaf node quantities (pass `logistics=0`).
2. Aggregate leaves by `type_id` across the whole tree (`_collect_leaves` recurses).
3. Pre-warm ESI cache: `_prefetch_ore_orders()` fetches all compressed ore type_ids
   and their refining byproducts in parallel using `ThreadPoolExecutor(max_workers=20)`.
4. For each leaf, call `get_ore_sources_for_mineral(type_id)` (returns compressed ores only).
5. For each compressed ore candidate:
   ```
   minerals_per_batch = floor(max_minerals_per_batch × reprocessing_yield)
   batches            = ceil(mineral_qty / minerals_per_batch)
   ore_units          = batches × portionSize
   ore_unit_price     = get_station_fill_price(...)    # None → skip this ore
   ```
6. Compute logistics-adjusted byproduct credit:
   ```
   for each refining output of this ore:
       output_qty = floor(output.quantity × yield) × batches
       if output is the target mineral:
           leftover_qty = output_qty - mineral_qty     # excess from ceil rounding
       else:
           leftover_qty = output_qty                   # full byproduct amount
       if leftover_qty <= 0: continue
       buy_price        = jita_buy_price(out_type_id, leftover_qty)
       logistics_per_unit = vol_per_unit × leftover_logistics_isk_per_m3
       net_credit_per_unit = max(0, buy_price - logistics_per_unit)
       byproduct_credit += net_credit_per_unit × leftover_qty
   ```
   Ores with bulky byproducts (large m³ per unit) become less attractive when the
   haul rate is high, because the logistics cost erodes their byproduct credit.
7. `effective_cost = ore_unit_price × ore_units + refining_fee − byproduct_credit`
8. Append candidate; after all ores processed, sort list by `effective_cost` ascending.
9. Items with no ore source go into `direct_items` in both paths.

### 10.4 True Global Leftover — `_compute_global_leftover`

`_compute_global_leftover(ore_items, leaves, leftover_logistics_isk_per_m3, reprocessing_yield)`
sums ALL refining outputs across ALL chosen ores, subtracts required quantities, and prices
the true surplus at Jita buy minus haul cost:

```python
total_produced: dict[int, (name, qty)] = {}
for each chosen ore_item:
    batches = ore_item.quantity // portionSize
    for each refining output:
        output_qty = floor(quantity × yield) × batches
        total_produced[out_type_id] += output_qty

leftover_items = []
for type_id, (name, qty_produced) in total_produced.items():
    surplus = qty_produced - leaves.get(type_id, 0)
    if surplus <= 0: continue
    gross_isk    = jita_buy_price(type_id, surplus) × surplus
    logistics    = vol_per_unit × surplus × leftover_logistics_isk_per_m3
    net_isk      = max(0, gross_isk - logistics)
    leftover_items.append(LeftoverItem(... volume_m3, logistics_isk, net_isk))

leftover_net_isk = max(0, sum(gross) - sum(logistics))
```

The per-item `byproduct_credit` in `ore_items` may be slightly over-counted (see section
15 gotcha 8). `leftover_net_isk` from `_compute_global_leftover` is the authoritative figure
used for all frontend cost comparisons.

### 10.5 Two-Pass Optimisation for `max_leftover_isk`

**Pass 1**: select `candidates[0]` (lowest logistics-adjusted effective_isk) for each mineral.
Compute global leftover with `_compute_global_leftover`.

**Pass 2** (only when `max_leftover_isk` is set and `leftover_net_isk > max_leftover_isk`):

```
repeat up to 30 times:
    if leftover_net_isk <= max_leftover_isk: break

    for each mineral:
        for each alternative ore (not current selection):
            test_items = swap this ore in
            _, _, _, test_net = _compute_global_leftover(test_items, ...)
            reduction = leftover_net_isk - test_net

    apply the swap with the greatest reduction
    recompute global leftover
    if no swap reduces leftover: break   # constraint is impossible
```

Each iteration picks the single best swap across all minerals. After 30 iterations (or
when no improving swap exists) the loop exits. If `leftover_net_isk` still exceeds the
limit, `leftover_constraint_met = false` is returned.

**Constraint-impossible fallback**: the frontend detects `leftover_constraint_met === false`,
shows a short muted note ("Leftover limit impossible with compressed ore — direct buy price
used"), and falls back to direct-buy costs in the breakdown. The leftover section is hidden.

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
- **Calculate**: fires `/build-cost` first; renders build results immediately when it
  returns (~2s). `/compare-material-source` is fired concurrently but awaited separately —
  the compare section populates once it resolves. This prevents large items (Keepstar,
  Avatar) from stalling the UI for 10–15s on a cold ESI cache. If compare fails, build
  results remain intact. `setLoading(false)` is called as soon as build-cost returns, not
  after compare.
- **Cost breakdown table**: shows each component as ISK and % of total. When
  `leftover_net_isk > 0` AND `leftover_constraint_met`, switches to "Net total cost"
  headline and uses ore path costs (`co.total_isk + co.refining_fee`) as material cost
  basis. Net total = ore + fees − `leftover_net_isk`. If `leftover_constraint_met` is
  false, a muted note appears below the headline and direct-buy costs are used instead.
- **Blueprint copies section**: populated by `renderBpcList()` from `bpo_list`. Columns:
  blueprint, activity, total runs, copies, runs/copy, copy cost, BPC/BPO toggle.
  - Copyable blueprints: calculated in-game copy job fee
  - Drop-only BPCs (Triglavian, SoCT) and faction/navy/officer hulls (`is_copyable=false`): user input field
  - Reactions: static "BPO" label, no toggle
  - Each non-reaction row has an Apple-style segmented BPC/BPO toggle (separate rightmost column). Toggling to BPO zeroes the row's contribution to the total.
  - Footer live-sums all active (BPC-state) copy costs including user inputs.
  - BPC total is included as a "Blueprint copies" row in the cost breakdown table and added to `total_cost`. Toggles re-render the breakdown headline immediately via `_activeBpcTotal` / `_lastBuild` pattern.
  - `_bpoToggled` set and `_userBpcCosts` dict reset when the user selects a different item.
- **Compare table**: Direct buy vs compressed ore side-by-side. Shows:
  - Net material cost (ore purchase − `leftover_net_isk`)
  - Ore purchase cost (sub-row)
  - Leftover credit sub-row = `−leftover_net_isk`; when logistics > 0, two further
    indented sub-rows show gross sell value and `−haul cost`
  - Refining fee
  - Net total = `co.total_isk + co.refining_fee − leftover_net_isk`
  - Volume to haul (compressed ore m³) and volume after refining
  Winning path determined by net total; winner highlighted green.
- **Ore breakdown table**: one row per ore item. Shows ore cost, refining fee, byproduct
  credit (logistics-adjusted), effective ISK, volume m³. `direct_items` shown as "direct buy".
- **Shopping list section**: two tabs, EVE multibuy format (`Item Name x Quantity` per line).
  - Tab 1 "Compressed ore": ores aggregated + `-- Sell leftovers --` separator + leftover items
  - Tab 2 "Direct buy": minerals bought directly
  - Tab labels show upfront purchase cost (`co.total_isk` / `db.total_isk`)
  - Cheaper badge (✓) determined by `co.total_isk + co.refining_fee − leftover_net_isk` vs direct
  - Defaults to cheaper tab; Copy button with "Copied!" flash
- **Leftover materials section**: hidden when `leftover_constraint_met` is false.
  When visible: shows Material / Quantity / Jita buy / Total ISK (no-logistics layout) or
  Material / Quantity / Jita buy / Volume / Haul cost / Net value (logistics layout).
  Footer shows gross value, haul cost, and net leftover credit when logistics > 0.
- **BOM tree**: recursive DOM rendering. Click any parent node to collapse/expand children.
  Shows quantity, total cost, and BPC info per node.
- **Enter key**: triggers Calculate when focus is on an input (unless an autocomplete
  dropdown is open).

### Collapsible sections

Every result section (`section-title` + `section-body` inside `.collapsible`) can be
folded by clicking the title. `toggleSection(id)` adds/removes the `collapsed` CSS class;
`.collapsible.collapsed > .section-body { display: none }` hides the body. A ▾/▸ chevron
updates on toggle. Cost badges (`.sec-cost`) in the title show key values when collapsed:
- `#total-cost` — headline total (always visible in title)
- `#bpc-cost-badge` — live BPC total (updated by `_refreshBpcTotal`)
- `#leftover-cost-badge` — net leftover credit (set by `renderLeftover`)

### Cache-busting

`app.js` is served with a `?v=N` query string in `index.html`. Increment `N` on every
change to force browsers to fetch the updated file.

### Form inputs

- **Buy from**: `jita_sell | jita_buy | amarr_sell | amarr_buy`
- **FW Bonus**: `None (0) | Level 1–5`
- **Structure bonus**: `EC no rig (1%) | EC + T1 rig (4%) | EC + T2 rig (5.5%)`
- **Build T1 hull**: checkbox, default checked. Unchecked sends `build_t1_hull: false`
  to both endpoints, which prices the T1 hull off market instead of recursing into it.
- **Max leftover value (ISK)**: optional number input; empty = no limit. Sent only to
  `/compare-material-source`, not to `/build-cost`.
- **Leftover haul cost (ISK/m³)**: ISK per m³ to move surplus minerals to Jita. Sent
  only to `/compare-material-source`.

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
    build_t1_hull: bool = True      # True = manufacture T1 hull; False = buy from market
```

`region_id` was removed — derived from `material_source` via `esi.STATION_REGION`.
`build_t1_hull`: when building a T2 ship, the T1 hull is a direct blueprint ingredient.
`True` (default) recurses into it and manufactures it from minerals. `False` treats it
as a leaf and prices it off market. See section 8.7.

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
    total_isk: ISK           # raw ore purchase price
    refining_fee: ISK
    byproduct_credit: ISK    # logistics-adjusted per-ore credit (see section 10.3 step 6)
    effective_isk: ISK       # total_isk + refining_fee − byproduct_credit
    volume_m3: ISK           # compressed ore volume (haul to refinery)
    refined_m3: ISK          # total mineral volume produced after refining

class LeftoverItem(BaseModel):
    type_id: int; name: str
    quantity: int; buy_price: ISK; total_isk: ISK
    volume_m3: ISK = 0       # total m³ of this surplus item
    logistics_isk: ISK = 0   # haul cost = volume_m3 × leftover_logistics_isk_per_m3
    net_isk: ISK = 0         # total_isk − logistics_isk (≥ 0)

class CompressedOrePath(BaseModel):
    total_isk: ISK            # raw ore purchase price across all ore items
    effective_isk: ISK        # sum of per-ore effective_isk (used for ore selection display)
    total_m3: ISK             # compressed ore volume + non-mineral direct volume
    refined_total_m3: ISK     # total mineral volume after refining + non-mineral direct
    refining_fee: ISK
    ore_items: list[CompressedOreItem]
    direct_items: list[DirectBuyItem]
    leftover_items: list[LeftoverItem] = []
    leftover_total_isk: ISK = 0        # gross Jita buy value of all surplus
    leftover_logistics_isk: ISK = 0    # total haul cost for all surplus
    leftover_net_isk: ISK = 0          # leftover_total_isk − leftover_logistics_isk
    leftover_constraint_met: bool = True  # False when max_leftover_isk unachievable

class CompareMaterialSourceRequest(BaseModel):
    # All BuildCostRequest fields, plus:
    build_t1_hull: bool = True      # inherited behaviour — see BuildCostRequest
    reprocessing_yield: float = 0.876
    reprocessing_rate: Decimal = Decimal("0.02")
    refinery_bonus: float = 0.0
    leftover_logistics_isk_per_m3: Decimal = Decimal("0")
    max_leftover_isk: Decimal | None = None
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

11. **Always use `leftover_net_isk` for comparisons**, never `leftover_total_isk`. The
    gross value ignores the haul cost and overstates the credit when logistics > 0.

12. **`leftover_constraint_met = false` means fall back entirely to direct buy**. The
    ore path numbers are still present in the response (for the compare table), but the
    cost breakdown and headline must use direct-buy costs. Do not show the leftover section
    in this state — it would imply the ore path is active when it is not.

13. **`ceil_qty` uses the batch formula, not per-run ceiling**. EVE applies ONE ceiling
    across the full batch: `max(runs, ceil(base × runs × (1-ME) × (1-sb)))`. An older
    per-run formula `max(1, ceil(base × (1-ME) × (1-sb))) × runs` over-counts when
    fractional rounding accumulates differently across a large batch. The batch formula
    is implemented in `app/core/utils.py`.

14. **`runs_needed` must be computed per BOM node, not from top-level `req.runs`**. Each
    sub-job has its own run count: `runs_needed = ceil(quantity_needed / qty_per_run)`.
    Using `req.runs` for sub-jobs gives the wrong EIV and wrong material quantities for
    any intermediate with `qty_per_run > 1` (all reaction products).

15. **Unpublished "Test Reaction Blueprint" stubs**. Some reaction products have two
    blueprint rows: a published "Reaction Formula" (qty ≈ 10,000/run) and an unpublished
    "Test Reaction Blueprint" (qty ≈ 20/run). Without `AND bt.published = 1` the test
    stub can win `LIMIT 1`, producing 500× too many runs and cascading into millions of
    moon material inputs. Always filter `published = 1` and order by `quantity DESC`.

16. **Moon material group ID is 427, not 711**. Group 711 is Harvestable Cloud (gas
    Fullerites). Moon materials (Atmospheric Gases, Evaporite Deposits, Tungsten, Platinum,
    Cobalt, etc.) live in group 427. Using the wrong group ID causes moon material nodes
    to recurse instead of being treated as market leaves.

17. **`compare.py` must unpack the `_compute_flat_bom` tuple**. `_compute_flat_bom` returns
    `(leaf_demands, node_runs)`. `compare.py` must unpack with `flat_bom, _ = ...`.
    Using the tuple directly causes `AttributeError: 'tuple' object has no attribute 'items'`
    and a 500 error on the compare endpoint, hiding the shopping list and ore comparison sections.

18. **Triglavian/SoCT blueprints have `activityID=5` in the SDE but no NPC BPO**.
    `has_copy_activity` must join `invTypes` and require `marketGroupID IS NOT NULL`.
    Without this, Zirnitra, Leshak, Gnosis, etc. are incorrectly marked copyable.

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
