from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from typing import List, Dict, Optional
import asyncio
import logging
from datetime import datetime

from core import AdvancedSearchEngine, SocialMediaSearcher
from models import SearchResult

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/search")
async def search_person(
    searchName: str = Query(..., description="Name to search for"),
    includeSocial: bool = Query(True, description="Include social media search"),
    maxResults: int = Query(20, description="Maximum results to return")
):
    """Enhanced search endpoint with Google Custom Search API and comprehensive error handling"""
    
    # Input validation
    if not searchName or len(searchName.strip()) < 2:
        return {
            "error": "Search query too short",
            "message": "Search query must be at least 2 characters long"
        }

    query = searchName.strip()
    max_results = min(maxResults, 50)  # Limit to 50 results max

    logger.info(f"Search request - Query: '{query}', Max results: {max_results}, Include social: {includeSocial}")

    try:
        # Initialize search engine
        async with AdvancedSearchEngine() as search_engine:
            # Perform web search
            web_results = await search_engine.search_multiple_engines(query, max_results)
            
            # Check if we got any results
            if not web_results:
                logger.warning(f"No web results found for query: {query}")
                return {
                    "message": f"No relevant web results found for '{query}'",
                    "query": query,
                    "total_results": 0,
                    "webpages": [],
                    "suggestions": [
                        "Check the spelling of the name",
                        "Try using the full name instead of a nickname",
                        "Include middle name or initial if known",
                        "Try alternative spellings or variations of the name",
                        "Use quotes around the exact name: \"John Smith\"",
                        "Add additional context like location or profession"
                    ]
                }

            # Format the main response (matching original Flask structure)
            result = {
                "query": query,
                "total_results": len(web_results),
                "timestamp": datetime.now().isoformat(),
                "webpages": []
            }

            # Process web results for frontend consumption
            for r in web_results:
                webpage_result = {
                    "url": r.url,
                    "title": r.title or "Untitled",
                    "description": r.description or "No description available",
                    "domain": r.domain,
                    "relevance_score": r.relevance_score
                }
                result["webpages"].append(webpage_result)

            logger.info(f"Found {len(web_results)} web results for '{query}'")

            # Add social media search if requested
            if includeSocial:
                logger.info(f"Searching for social media profiles for: {query}")
                try:
                    social_searcher = SocialMediaSearcher(search_engine)
                    social_media_results = await social_searcher.find_social_accounts(query)

                    result["social_media"] = {}
                    total_social_results = 0

                    # Process social media results for each platform
                    for platform, accounts in social_media_results.items():
                        platform_results = []
                        for acc in accounts:
                            social_result = {
                                "url": acc.url,
                                "title": acc.title or f"{platform.capitalize()} Profile",
                                "description": acc.description or f"Potential {platform.capitalize()} profile for {query}",
                                "username": acc.username or "Unknown",
                                "relevance_score": acc.relevance_score,
                                "platform": platform
                            }
                            platform_results.append(social_result)

                        result["social_media"][platform] = platform_results
                        total_social_results += len(platform_results)

                    result["total_social_results"] = total_social_results
                    logger.info(f"Found {total_social_results} social media profiles across all platforms")

                except Exception as social_error:
                    logger.error(f"Social media search failed: {social_error}")
                    # Don't fail the entire request if social search fails
                    result["social_media"] = {}
                    result["total_social_results"] = 0
                    result["social_search_error"] = "Social media search temporarily unavailable"
            else:
                result["social_media"] = {}
                result["total_social_results"] = 0

            # Add search metadata with Google Custom Search API info
            result["search_metadata"] = {
                "engines_used": ["google_custom_search", "google_direct"],
                "primary_engine": "google_custom_search",
                "google_custom_search_available": True,
                "search_time": datetime.now().isoformat(),
                "include_social_media": includeSocial
            }

            # Log successful completion
            total_results = len(result["webpages"]) + result.get("total_social_results", 0)
            logger.info(f"Search completed successfully. Total results: {total_results}")

            return result

    except asyncio.TimeoutError:
        logger.error(f"Search timeout for query: {query}")
        return {
            "error": "Search timeout",
            "message": "The search request took too long to complete. Please try again.",
            "query": query,
            "suggestions": [
                "Try a more specific search term",
                "Check your internet connection",
                "Try again in a few moments"
            ]
        }

    except Exception as e:
        logger.error(f"Search error for query '{query}': {e}")
        return {
            "error": "Search failed",
            "message": "An internal error occurred during search. Please try again.",
            "query": query
        }
