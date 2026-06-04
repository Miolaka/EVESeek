import time
from decimal import Decimal
from typing import Any, Optional
import httpx

ESI_BASE = "https://esi.evetech.net/latest"

# TTL in seconds
_TTL_MARKET = 300       # 5 min — regional market orders
_TTL_COST_INDEX = 3600  # 1 hr  — system cost index
_TTL_ADJ_PRICE = 86400  # 24 hr — universe adjusted prices

_cache: dict[str, tuple[float, Any]] = {}


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
    """Fetch buy and sell orders for a type in a region."""
    key = f"orders:{region_id}:{type_id}"
    cached = _get(key)
    if cached is not None:
        return cached

    orders = _fetch(f"/markets/{region_id}/orders/", {"type_id": type_id, "order_type": "all"})
    _set(key, orders, _TTL_MARKET)
    return orders


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


def get_system_cost_index(system_id: int) -> dict[int, Decimal]:
    """Cost indices per activity for a solar system, cached 1hr."""
    key = f"cost_index:{system_id}"
    cached = _get(key)
    if cached is not None:
        return cached

    data = _fetch("/industry/systems/")
    for system in data:
        if system["solar_system_id"] == system_id:
            indices = {item["activity"]: Decimal(str(item["cost_index"])) for item in system["cost_indices"]}
            _set(key, indices, _TTL_COST_INDEX)
            return indices

    _set(key, {}, _TTL_COST_INDEX)
    return {}
