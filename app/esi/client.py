import time
from decimal import Decimal
from typing import Any, Optional
import httpx
from app.esi import disk_cache

ESI_BASE = "https://esi.evetech.net/latest"

# TTL in seconds
_TTL_MARKET = 300        # 5 min — regional market orders
_TTL_COST_INDEX = 86400  # 24 hr — system cost indices (full universe, cached as one key)
_TTL_ADJ_PRICE = 86400   # 24 hr — universe adjusted prices

_cache: dict[str, tuple[float, Any]] = {}

# Station → region mapping for known trade hubs
STATION_REGION: dict[int, int] = {
    60003760: 10000002,  # Jita IV-4 → The Forge
    60008494: 10000043,  # Amarr EFA → Domain
}


def _get(key: str) -> Optional[Any]:
    entry = _cache.get(key)
    if entry and time.time() < entry[0]:
        return entry[1]
    return None


def _set(key: str, value: Any, ttl: int) -> None:
    _cache[key] = (time.time() + ttl, value)


def _fetch(path: str, params: dict | None = None) -> Any:
    response = httpx.get(f"{ESI_BASE}{path}", params=params, timeout=10)
    response.raise_for_status()
    return response.json()


def get_market_orders(region_id: int, type_id: int) -> list[dict]:
    """Fetch all buy and sell orders for a type in a region (handles pagination).

    L1: in-memory dict (process lifetime).
    L2: SQLite disk cache at data/esi_cache.db (survives restarts, TTL 5 min).
    L3: ESI HTTP call.
    """
    key = f"orders:{region_id}:{type_id}"

    cached = _get(key)
    if cached is not None:
        return cached

    disk_hit = disk_cache.get(key)
    if disk_hit is not None:
        _set(key, disk_hit, _TTL_MARKET)
        return disk_hit

    url = f"{ESI_BASE}/markets/{region_id}/orders/"
    params: dict[str, Any] = {"type_id": type_id, "order_type": "all", "page": 1}

    first = httpx.get(url, params=params, timeout=10)
    first.raise_for_status()
    orders: list[dict] = first.json()

    total_pages = int(first.headers.get("X-Pages", 1))
    for page in range(2, total_pages + 1):
        params["page"] = page
        r = httpx.get(url, params=params, timeout=10)
        r.raise_for_status()
        orders.extend(r.json())

    _set(key, orders, _TTL_MARKET)
    disk_cache.put(key, orders, _TTL_MARKET)
    return orders


def get_station_fill_price(
    region_id: int,
    type_id: int,
    station_id: int,
    quantity: int,
    is_buy_order: bool,
) -> Optional[Decimal]:
    """Volume-weighted average fill price at a specific station for a given quantity.

    Sorts sell orders ascending (or buy orders descending) and fills greedily.
    Returns the weighted average over filled volume; None if station has no orders.
    """
    orders = get_market_orders(region_id, type_id)
    station_orders = [
        o for o in orders
        if o["location_id"] == station_id and o["is_buy_order"] == is_buy_order
    ]
    if not station_orders:
        return None

    station_orders.sort(key=lambda o: o["price"], reverse=is_buy_order)

    remaining = quantity
    total_isk = Decimal("0")
    total_filled = 0

    for order in station_orders:
        if remaining <= 0:
            break
        fill = min(remaining, order["volume_remain"])
        total_isk += Decimal(str(order["price"])) * fill
        total_filled += fill
        remaining -= fill

    if total_filled == 0:
        return None

    return total_isk / total_filled


def get_best_sell(region_id: int, type_id: int) -> Optional[Decimal]:
    """Lowest sell price in region."""
    orders = get_market_orders(region_id, type_id)
    sells = [o["price"] for o in orders if not o["is_buy_order"]]
    return Decimal(str(min(sells))) if sells else None


def get_best_buy(region_id: int, type_id: int) -> Optional[Decimal]:
    """Highest buy price in region."""
    orders = get_market_orders(region_id, type_id)
    buys = [o["price"] for o in orders if o["is_buy_order"]]
    return Decimal(str(max(buys))) if buys else None


def get_adjusted_prices() -> dict[int, Decimal]:
    """Universe-wide adjusted prices (EIV) from ESI, cached 24hr."""
    key = "adjusted_prices"
    cached = _get(key)
    if cached is not None:
        return cached

    data = _fetch("/markets/prices/")
    result = {item["type_id"]: Decimal(str(item["adjusted_price"])) for item in data if "adjusted_price" in item}
    _set(key, result, _TTL_ADJ_PRICE)
    return result


def get_adjusted_price(type_id: int) -> Optional[Decimal]:
    return get_adjusted_prices().get(type_id)


def _get_all_cost_indices() -> dict[int, dict[str, Decimal]]:
    """Fetch and cache all system cost indices as one daily snapshot."""
    key = "all_cost_indices"
    cached = _get(key)
    if cached is not None:
        return cached
    data = _fetch("/industry/systems/")
    result: dict[int, dict[str, Decimal]] = {}
    for system in data:
        result[system["solar_system_id"]] = {
            item["activity"]: Decimal(str(item["cost_index"]))
            for item in system["cost_indices"]
        }
    _set(key, result, _TTL_COST_INDEX)
    return result


def get_system_cost_index(system_id: int) -> dict[str, Decimal]:
    """Cost indices per activity for a solar system. Full universe cached once per day."""
    return _get_all_cost_indices().get(system_id, {})
