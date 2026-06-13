from app.data.sde import _conn
from typing import Any


def get_standup_me_rigs() -> list[dict[str, Any]]:
    """Return all published standup material-efficiency rigs with bonus data."""
    with _conn() as conn:
        rigs = conn.execute("""
            SELECT t.typeID, t.typeName
            FROM invTypes t
            WHERE t.published = 1
              AND (t.typeName LIKE '%Standup%Efficiency%'
                   OR t.typeName LIKE '%Standup%Reactor%Efficiency%')
            ORDER BY t.typeName
        """).fetchall()

        def ga(tid, attr_id):
            r = conn.execute(
                "SELECT COALESCE(valueFloat, valueInt) AS v "
                "FROM dgmTypeAttributes WHERE typeID=? AND attributeID=?",
                (tid, attr_id),
            ).fetchone()
            return r["v"] if r else None

        result = []
        for r in rigs:
            tid = r["typeID"]
            mat_mfg   = ga(tid, 2594)   # manufacturing material reduction
            mat_react = ga(tid, 2714)   # reaction material reduction
            base_me   = mat_mfg if mat_mfg is not None else mat_react
            if base_me is None or base_me == 0.0:
                continue   # skip time-only rigs
            hi   = ga(tid, 2355) or 1.0
            lo   = ga(tid, 2356) or 1.0
            nu   = ga(tid, 2357) or 1.0
            size = int(ga(tid, 1547) or 0)
            banned_hi = ga(tid, 1970) == 1.0

            result.append({
                "type_id":     tid,
                "name":        r["typeName"],
                "base_me_pct": abs(base_me),   # positive %, e.g. 2.0 means 2%
                "hi_mult":     hi,
                "lo_mult":     lo,
                "nu_mult":     nu,
                "rig_size":    size,
                "banned_hi":   banned_hi,
                "is_reaction": mat_react is not None and mat_react != 0.0,
            })
        return result
