# EVESeek

EVE Online manufacturing cost calculator. Computes full supply-chain build cost from raw materials to finished ship, replicating Eve's in-game formulas exactly.

## Features

- Full recursive BOM (Bill of Materials) — manufacturing, reactions, refining
- Accurate job fee calculation (manufacturing, reaction, refining)
- Configurable logistics cost (ISK per m³)
- Regional market prices via ESI
- Material efficiency (ME) support
- Faction Warfare bonus support
- Decimal precision for all ISK values

## Stack

- **Backend**: Python, FastAPI
- **Data**: Eve SDE (Fuzzwork SQLite + Hoboleaks JSON)
- **Market data**: ESI (Eve Swagger Interface)

## Setup

**1. Install dependencies**
```bash
mamba install fastapi uvicorn httpx -c conda-forge
```

**2. Download SDE data**
```bash
python scripts/download_fuzzwork_sde.py   # item types, volumes
python scripts/download_sde.py            # blueprint & industry data
```

**3. Run**
```bash
uvicorn app.main:app --reload
```

API docs available at `http://localhost:8000/docs`

## API

### `POST /api/v1/build-cost`

Calculates full build cost for an item.

```json
{
  "type_id": 28606,
  "region_id": 10000002,
  "system_id": 30000142,
  "me_level": 10,
  "structure_bonus": 0.04,
  "fw_level": 0,
  "runs": 1,
  "material_source": "jita_sell",
  "logistics_cost_isk_per_m3": 800
}
```

### `GET /api/v1/search?q=paladin`

Search items by name.

## Project Status

- [x] SDE download scripts
- [x] FastAPI scaffold, ESI client, SDE query layer
- [ ] BOM engine (recursive)
- [ ] Cost calculators
- [ ] Frontend UI
