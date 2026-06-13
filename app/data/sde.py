import sqlite3
import threading
from functools import lru_cache
from pathlib import Path
from typing import Optional

_DB_PATH = Path(__file__).parent.parent.parent / "data" / "eve.db"

# Activity IDs
ACTIVITY_MANUFACTURING = 1
ACTIVITY_REACTION = 11

_local = threading.local()


def _conn() -> sqlite3.Connection:
    conn = getattr(_local, "conn", None)
    if conn is None:
        conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        _local.conn = conn
    return conn


@lru_cache(maxsize=4096)
def get_type(type_id: int) -> Optional[sqlite3.Row]:
    """Return type metadata including packaged volume."""
    return _conn().execute(
        """
        SELECT t.typeID, t.typeName, t.groupID, t.mass, t.portionSize,
               COALESCE(v.volume, t.volume) AS volume
        FROM invTypes t
        LEFT JOIN invVolumes v ON t.typeID = v.typeID
        WHERE t.typeID = ?
        """,
        (type_id,),
    ).fetchone()


@lru_cache(maxsize=4096)
def get_blueprint_for_product(product_type_id: int) -> Optional[sqlite3.Row]:
    """Return the blueprint that produces the given product type."""
    return _conn().execute(
        """
        SELECT p.typeID AS blueprint_type_id, p.activityID, p.quantity
        FROM industryActivityProducts p
        JOIN invTypes bt ON p.typeID = bt.typeID
        WHERE p.productTypeID = ?
          AND p.activityID IN (?, ?)
          AND bt.published = 1
        ORDER BY p.quantity DESC
        LIMIT 1
        """,
        (product_type_id, ACTIVITY_MANUFACTURING, ACTIVITY_REACTION),
    ).fetchone()


@lru_cache(maxsize=4096)
def get_activity_materials(
    blueprint_type_id: int, activity_id: int
) -> list[sqlite3.Row]:
    """Return input materials for a blueprint activity."""
    return _conn().execute(
        """
        SELECT materialTypeID, quantity
        FROM industryActivityMaterials
        WHERE typeID = ? AND activityID = ?
        """,
        (blueprint_type_id, activity_id),
    ).fetchall()


def get_activity_time(blueprint_type_id: int, activity_id: int) -> int:
    """Return base job time in seconds."""
    row = _conn().execute(
        "SELECT time FROM industryActivity WHERE typeID = ? AND activityID = ?",
        (blueprint_type_id, activity_id),
    ).fetchone()
    return row["time"] if row else 0


@lru_cache(maxsize=1024)
def get_group(group_id: int) -> Optional[sqlite3.Row]:
    """Return group metadata (groupID, categoryID, groupName)."""
    return _conn().execute(
        "SELECT groupID, categoryID, groupName FROM invGroups WHERE groupID = ?",
        (group_id,),
    ).fetchone()


def get_refining_outputs(type_id: int) -> list[sqlite3.Row]:
    """Return minerals/materials produced by refining this type."""
    return _conn().execute(
        "SELECT materialTypeID, quantity FROM invTypeMaterials WHERE typeID = ?",
        (type_id,),
    ).fetchall()


def get_ore_sources_for_mineral(mineral_type_id: int) -> list[sqlite3.Row]:
    """Return all compressed ores that yield this mineral when refined."""
    return _conn().execute(
        """
        SELECT t.typeID AS ore_type_id, t.typeName, t.portionSize,
               m.quantity AS max_minerals_per_batch
        FROM invTypeMaterials m
        JOIN invTypes t ON t.typeID = m.typeID
        JOIN invGroups g ON t.groupID = g.groupID
        WHERE m.materialTypeID = ?
          AND t.published = 1
          AND g.categoryID = 25
          AND t.typeName LIKE 'Compressed %'
        ORDER BY t.typeName
        """,
        (mineral_type_id,),
    ).fetchall()


@lru_cache(maxsize=4096)
@lru_cache(maxsize=4096)
def has_copy_activity(blueprint_type_id: int) -> bool:
    """Return True if this blueprint can be copied via a BPO.

    Requires both activityID=5 AND a non-null marketGroupID on the blueprint.
    Drop-only BPCs (Triglavian ships, SoCT ships, etc.) have activityID=5 in
    the SDE but no marketGroupID — they have no NPC-purchasable BPO to copy.
    """
    row = _conn().execute(
        """SELECT 1 FROM industryActivity ia
           JOIN invTypes t ON t.typeID = ia.typeID
           WHERE ia.typeID = ? AND ia.activityID = 5
             AND t.marketGroupID IS NOT NULL""",
        (blueprint_type_id,),
    ).fetchone()
    return row is not None


@lru_cache(maxsize=4096)
def get_max_production_limit(blueprint_type_id: int) -> int:
    """Return maxProductionLimit for a blueprint (max runs per BPC copy)."""
    row = _conn().execute(
        "SELECT maxProductionLimit FROM industryBlueprints WHERE typeID = ?",
        (blueprint_type_id,),
    ).fetchone()
    return row["maxProductionLimit"] if row else 1


def search_solar_systems(query: str, limit: int = 20) -> list[sqlite3.Row]:
    """Search solar systems by name."""
    return _conn().execute(
        """
        SELECT solarSystemID, solarSystemName, security
        FROM mapSolarSystems
        WHERE solarSystemName LIKE ?
        ORDER BY solarSystemName
        LIMIT ?
        """,
        (f"%{query}%", limit),
    ).fetchall()


@lru_cache(maxsize=4096)
def get_meta_group(type_id: int) -> Optional[int]:
    row = _conn().execute(
        "SELECT metaGroupID FROM invMetaTypes WHERE typeID = ?",
        (type_id,),
    ).fetchone()
    return row["metaGroupID"] if row else 1


def search_types(query: str, limit: int = 20) -> list[sqlite3.Row]:
    """Search item types by name (for the UI search box)."""
    return _conn().execute(
        """
        SELECT typeID, typeName
        FROM invTypes
        WHERE typeName LIKE ? AND published = 1
        ORDER BY typeName
        LIMIT ?
        """,
        (f"%{query}%", limit),
    ).fetchall()
