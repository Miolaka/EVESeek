from functools import lru_cache
from app.data import sde

ACTIVITY_MANUFACTURING = 1
ACTIVITY_REACTION = 11

# Capital ship group IDs
_CAP_SHIP_GROUPS = frozenset({30, 485, 547, 659, 883, 902, 1538, 4594})
# Large ship group IDs (battleships + T2 variants)
_LARGE_SHIP_GROUPS = frozenset({27, 898, 900})  # Battleship, Black Ops, Marauder
# Medium ship group IDs
_MED_SHIP_GROUPS = frozenset({
    26, 358, 380, 419, 463, 540, 541, 832, 833, 894, 906, 963,
    1201, 1202, 1527, 1534, 1972,
})

_FUEL_BLOCK_GROUP = 1136

# Reaction product group IDs
_COMP_REACT_GROUPS = frozenset({428, 429, 4096})
_HYB_REACT_GROUPS  = frozenset({974})
_BIO_REACT_GROUPS  = frozenset({712})


@lru_cache(maxsize=8192)
def get_slot(type_id: int, activity_id: int) -> str:
    """Return the Ravworks-style activity slot name for this item."""
    if activity_id == ACTIVITY_REACTION:
        t = sde.get_type(type_id)
        if t:
            g = t["groupID"]
            if g in _HYB_REACT_GROUPS:  return "hyb_react"
            if g in _BIO_REACT_GROUPS:  return "bio_react"
        return "comp_react"

    t = sde.get_type(type_id)
    if not t:
        return "equipment"

    group = sde.get_group(t["groupID"])
    if not group:
        return "equipment"

    cat_id = group["categoryID"]
    gid    = t["groupID"]

    if gid == _FUEL_BLOCK_GROUP:
        return "fuel_blocks"

    if cat_id == 6:  # Ships
        if gid in _CAP_SHIP_GROUPS:
            return "cap_ship"
        meta = sde.get_meta_group(type_id) or 1
        t2 = (meta == 2)
        if gid in _LARGE_SHIP_GROUPS:
            return "adv_large_ship" if t2 else "basic_large_ship"
        if gid in _MED_SHIP_GROUPS:
            return "adv_med_ship" if t2 else "basic_med_ship"
        return "adv_small_ship" if t2 else "basic_small_ship"

    if cat_id == 8:   return "ammo"
    if cat_id == 18:  return "drones"

    if cat_id == 17:  # Construction Components
        meta = sde.get_meta_group(type_id) or 1
        t2   = (meta == 2)
        name = t["typeName"] if t else ""
        if "Capital" in name:
            return "cap_adv_comp" if t2 else "cap_comp"
        return "adv_comp" if t2 else "equipment"

    if cat_id in {65, 40}:
        return "structure"

    return "equipment"
