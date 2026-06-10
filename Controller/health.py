from fastapi import APIRouter
from datetime import datetime
import logging
from apify_scraper import test_apify_setup

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/apify/health")
async def apify_health_check():
    """Health check endpoint specifically for APIFY functionality"""
    try:
        test_result = await test_apify_setup()

        health_status = {
            "status": "healthy" if test_result.get("test_url_success", False) else "degraded",
            "timestamp": datetime.now().isoformat(),
            "apify_status": test_result,
            "optimization_status": {
                "concurrent_processing": True,
                "concurrent_fetch_limit": 10,
                "concurrent_gpt_limit": 5,
                "gpt_rate_limit_delay": 0.2,
                "url_fetch_timeout": 30
            }
        }

        # If APIFY is not working, it's degraded but not failed (fallback available)
        if not test_result.get("apify_available", False):
            health_status["status"] = "degraded"
            health_status["message"] = "APIFY not available, using fallback scraper"
        elif not test_result.get("api_key_configured", False):
            health_status["status"] = "degraded"
            health_status["message"] = "APIFY API key not configured"
        elif not test_result.get("test_url_success", False):
            health_status["status"] = "degraded"
            health_status["message"] = "APIFY test failed, using fallback scraper"

        return health_status

    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }

@router.get("/apify/test")
async def apify_test():
    """Test APIFY configuration and return detailed results"""
    try:
        test_result = await test_apify_setup()
        
        return {
            "test_results": test_result,
            "timestamp": datetime.now().isoformat(),
            "test_status": "completed"
        }

    except Exception as e:
        logger.error(f"APIFY test error: {e}")
        return {
            "test_results": {"error": str(e)},
            "timestamp": datetime.now().isoformat(),
            "test_status": "failed"
        }
