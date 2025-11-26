import os
import time
import asyncio
import aiohttp
from aiohttp import ClientSession, ClientTimeout
import json
from fastapi import FastAPI, HTTPException, Query, Body, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from math import exp
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import OpenAI
import ast
import re
import requests
import random
from urllib.parse import quote_plus, urlparse, parse_qs, urljoin
import itertools
from collections import defaultdict
from typing import List, Dict, Set, Tuple, Optional
from bs4 import BeautifulSoup, Comment
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib
import csv
from io import StringIO
import base64
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from difflib import SequenceMatcher
from apify_scraper import APIfyCheerioScraper, APIfyScraperManager, test_apify_setup
from dotenv import load_dotenv
import uvicorn

# Try to import enhanced packages
try:
    import cloudscraper
    from fake_useragent import UserAgent
    ENHANCED_SCRAPING = True
except ImportError:
    ENHANCED_SCRAPING = False
    print("Warning: cloudscraper or fake_useragent not installed. Using basic scraping.")

# Search APIs removed - using only Google Custom Search API

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="PrivacyGuard API",
    description="PII Detection and Privacy Protection API - FastAPI Version",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Configuration - ALLOW ALL ORIGINS FOR TESTING
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for testing
    allow_credentials=False,  # Must be False when using "*"
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Load environment variables
print("🔄 Loading environment variables...")
result = load_dotenv()
print(f"📄 load_dotenv() result: {result}")

# Initialize OpenAI client
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

client = OpenAI(api_key=api_key)

# User agent logic moved to core.py for better modularity

# OPTIMIZED: Configuration for concurrent processing
CONCURRENT_FETCH_LIMIT = 10  # Max concurrent URL fetches
CONCURRENT_GPT_LIMIT = 5  # Max concurrent GPT calls
GPT_RATE_LIMIT_DELAY = 0.2  # Delay between GPT calls (seconds)
URL_FETCH_TIMEOUT = 30  # Timeout for URL fetching (seconds)

@dataclass
class SearchResult:
    """Data class for search results"""
    url: str
    title: str
    description: str
    domain: str
    relevance_score: int = 0
    content_type: str = "webpage"
    platform: str = ""
    username: str = ""

@dataclass
class ExtractionResult:
    """Data class for extraction results"""
    url: str
    pii_found: List[Dict]
    confidence_score: float
    processing_time: float
    success: bool
    error: Optional[str] = None

# Import all the existing classes and functions from the original file
# We'll copy the core logic here to avoid circular imports

# Import routes
from routes import search, extract, health, performance

# Include routers
app.include_router(search.router, tags=["search"])
app.include_router(extract.router, tags=["extract"])
app.include_router(health.router, tags=["health"])
app.include_router(performance.router, tags=["performance"])

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "PrivacyGuard API - FastAPI Version",
        "version": "2.0.0",
        "docs": "/docs",
        "redoc": "/redoc",
        "status": "running"
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "framework": "FastAPI",
        "version": "2.0.0"
    }

# Root-level endpoints for backward compatibility with original Flask version
@app.get("/search")
async def search_person_root(
    searchName: str = Query(..., description="Name to search for"),
    parameters: str = Query(False, description="Search parameters"),
    includeSocial: bool = Query(True, description="Include social media search"),
    maxResults: int = Query(25, description="Maximum results to return")
):
    """Root-level search endpoint for backward compatibility"""
    from routes.search import search_person
    return await search_person(searchName, parameters, includeSocial, maxResults)

@app.post("/extract")
async def extract_pii_root(request: dict):
    """Root-level extract endpoint for backward compatibility"""
    from routes.extract import extract_pii
    from models import ExtractRequest
    
    # Convert dict to ExtractRequest
    extract_request = ExtractRequest(
        searchName=request.get('searchName', ''),
        selectedUrls=request.get('selectedUrls', []),
        selectedSocial=request.get('selectedSocial', [])
    )
    return await extract_pii(extract_request)

@app.get("/apify/health")
async def apify_health_root():
    """Root-level APIFY health endpoint for backward compatibility"""
    from routes.health import apify_health_check
    return await apify_health_check()

@app.get("/apify/test")
async def apify_test_root():
    """Root-level APIFY test endpoint for backward compatibility"""
    from routes.health import apify_test
    return await apify_test()

@app.get("/performance/stats")
async def performance_stats_root():
    """Root-level performance stats endpoint for backward compatibility"""
    from routes.performance import performance_stats
    return await performance_stats()

if __name__ == "__main__":
    import socket
    
    port = int(os.environ.get("PORT", 5003))
    
    # Get local IP address
    def get_local_ip():
        try:
            # Connect to a remote server to get local IP
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                return s.getsockname()[0]
        except:
            return "0.0.0.0"
    
    local_ip = get_local_ip()
    
    print("\n" + "="*60)
    print("🚀 PrivacyGuard API - FastAPI Server Starting...")
    print("="*60)
    print(f"�� Server running on:")
    print(f"   • Local:   http://127.0.0.1:{port}")
    print(f"   • Network: http://{local_ip}:{port}")
    print(f"   • Docs:    http://127.0.0.1:{port}/docs")
    print(f"   • Health:  http://127.0.0.1:{port}/health")
    print("="*60)
    print("Press CTRL+C to stop the server")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=port)
