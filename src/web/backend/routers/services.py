"""
Services router — system health monitoring, Redis/Celery status, error logs.
"""

from fastapi import APIRouter

from web.backend.services.health_service import get_full_health, get_recent_errors

router = APIRouter()


@router.get("/health")
async def services_health():
    """Comprehensive health check across all subsystems."""
    return get_full_health()


@router.get("/errors")
async def services_errors():
    """Recent error log."""
    errors = get_recent_errors(30)
    return {"errors": errors, "count": len(errors)}
