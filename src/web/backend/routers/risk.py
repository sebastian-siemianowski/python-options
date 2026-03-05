"""
Risk router — risk dashboard, temperature readings, stress categories.
"""

from fastapi import APIRouter

router = APIRouter()


@router.get("/dashboard")
async def risk_dashboard():
    """
    Full risk dashboard JSON.
    
    Note: This is a heavy computation (calls all 3 risk modules).
    For real-time use, prefer the Celery task via POST /api/risk/compute.
    """
    from web.backend.services.risk_service import compute_risk_json
    return compute_risk_json()


@router.get("/summary")
async def risk_summary():
    """Quick temperature summary (combined + per-module)."""
    from web.backend.services.risk_service import compute_risk_json, get_risk_temperature_summary
    risk_json = compute_risk_json()
    return get_risk_temperature_summary(risk_json)
