from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from datetime import datetime
import logging
import psutil
import time

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/performance/stats")
async def performance_stats():
    """Get performance statistics and system information"""
    try:
        # Get system information
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Get process information
        process = psutil.Process()
        process_memory = process.memory_info()
        
        stats = {
            "optimization_status": "enabled",
            "concurrent_limits": {
                "url_fetching": 10,
                "gpt_processing": 5
            },
            "timeouts": {
                "url_fetch_timeout": 30,
                "gpt_rate_limit_delay": 0.2
            },
            "processing_strategy": {
                "url_processing": "concurrent_with_semaphore",
                "gpt_processing": "concurrent_batches_with_rate_limiting",
                "retry_strategy": "fail_fast_no_retries",
                "connection_pooling": "optimized"
            },
            "features": {
                "concurrent_url_fetching": True,
                "concurrent_gpt_processing": True,
                "apify_integration": True,
                "fallback_scraper": True,
                "smart_error_handling": True,
                "performance_monitoring": True
            }
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Performance stats error: {e}")
        return {"error": f"Failed to get performance stats: {str(e)}"}
