"""
Risk router — risk dashboard, temperature readings, stress categories.
"""

from fastapi import APIRouter

router = APIRouter()


@router.get("/dashboard")
async def risk_dashboard():
    """
    Full risk dashboard JSON with caching.
    
    Returns cached data if fresh (< 1 hour), otherwise computes.
    Use POST /refresh to force recomputation.
    """
    from web.backend.services.risk_service import compute_risk_json
    return compute_risk_json()


@router.get("/summary")
async def risk_summary():
    """Quick temperature summary (combined + per-module)."""
    from web.backend.services.risk_service import compute_risk_json, get_risk_temperature_summary
    risk_json = compute_risk_json()
    return get_risk_temperature_summary(risk_json)


@router.post("/refresh")
async def risk_refresh():
    """Force recompute risk dashboard (ignores cache)."""
    from web.backend.services.risk_service import compute_risk_json, get_risk_temperature_summary
    risk_json = compute_risk_json(force=True)
    summary = get_risk_temperature_summary(risk_json)
    return {"status": "refreshed", "summary": summary}
