import os
import time
import asyncio
import aiohttp
from aiohttp import ClientSession, ClientTimeout
import json
from flask import Flask, jsonify, request
from flask_cors import CORS
from math import exp
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import OpenAI
from dotenv import load_dotenv
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

# Try to import enhanced packages
try:
    import cloudscraper
    from fake_useragent import UserAgent

    ENHANCED_SCRAPING = True
except ImportError:
    ENHANCED_SCRAPING = False
    print("Warning: cloudscraper or fake_useragent not installed. Using basic scraping.")

# Try to import search APIs
try:
    import serpapi

    SERPAPI_AVAILABLE = True
except ImportError:
    SERPAPI_AVAILABLE = False

# Import existing googlesearch
try:
    from googlesearch import search as google_search

    GOOGLE_SEARCH_AVAILABLE = True
except ImportError:
    GOOGLE_SEARCH_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

load_dotenv()

# Initialize OpenAI client
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

client = OpenAI(api_key=api_key)

# Initialize user agent rotator
if ENHANCED_SCRAPING:
    ua = UserAgent()
else:
    # Fallback user agents
    USER_AGENTS = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    ]

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
    source: str
    content: List[str]
    metadata: Dict[str, str]
    success: bool = True
    error: Optional[str] = None


# OPTIMIZED: Missing utility functions implementation
def detect_platform_from_url(url: str) -> str:
    """Detect social media platform from URL"""
    url_lower = url.lower()

    if 'facebook.com' in url_lower or 'fb.com' in url_lower:
        return 'facebook'
    elif 'twitter.com' in url_lower or 'x.com' in url_lower:
        return 'twitter'
    elif 'instagram.com' in url_lower:
        return 'instagram'
    elif 'linkedin.com' in url_lower:
        return 'linkedin'
    elif 'tiktok.com' in url_lower:
        return 'tiktok'
    elif 'youtube.com' in url_lower or 'youtu.be' in url_lower:
        return 'youtube'
    elif 'snapchat.com' in url_lower:
        return 'snapchat'
    elif 'pinterest.com' in url_lower:
        return 'pinterest'
    elif 'reddit.com' in url_lower:
        return 'reddit'
    else:
        return 'unknown'


def extract_username_from_url(url: str) -> str:
    """Extract username from social media URL"""
    try:
        parsed = urlparse(url)
        path = parsed.path.strip('/')
        platform = detect_platform_from_url(url)

        if platform == 'facebook':
            if 'profile.php' in url:
                return parse_qs(parsed.query).get('id', [''])[0]
            if 'people/' in path:
                return path.split('people/')[1].split('/')[0]
            return path.split('/')[0] if path else ''

        elif platform == 'twitter':
            return path.split('/')[0] if path else ''

        elif platform == 'instagram':
            return path.split('/')[0] if path else ''

        elif platform == 'linkedin':
            if '/in/' in path:
                return path.split('/in/')[1].split('/')[0]
            elif '/pub/' in path:
                return path.split('/pub/')[1].split('/')[0]
            return ''

        elif platform == 'tiktok':
            if '@' in path:
                return path.replace('@', '').split('/')[0]
            elif '/user/' in path:
                return path.split('/user/')[1].split('/')[0]
            return path.split('/')[0] if path else ''

        elif platform == 'youtube':
            for prefix in ['/c/', '/@', '/channel/', '/user/']:
                if prefix in path:
                    return path.split(prefix)[1].split('/')[0]
            return ''

        return path.split('/')[0] if path else ''

    except Exception as e:
        logger.error(f"Error extracting username from {url}: {e}")
        return ''


class NameMatcher:
    """Enhanced name matching for better recognition of variations"""

    @staticmethod
    def normalize_name(name: str) -> str:
        """Normalize name for better matching"""
        return re.sub(r'[^a-zA-Z\s]', '', name.lower().strip())

    @staticmethod
    def get_name_variations(name: str) -> List[str]:
        """Generate common variations of a name"""
        normalized = NameMatcher.normalize_name(name)
        parts = normalized.split()

        variations = [
            normalized,  # Original normalized
            name.lower(),  # Original case
            ' '.join(parts),  # Normalized with spaces
        ]

        # Add individual parts for partial matching (only meaningful parts)
        variations.extend([part for part in parts if len(part) > 2])

        # Add reverse order if multiple parts
        if len(parts) > 1:
            variations.append(' '.join(reversed(parts)))

        # Remove duplicates and return
        return list(set(filter(None, variations)))

    @staticmethod
    def calculate_name_similarity(target_name: str, text: str) -> float:
        """Calculate similarity between target name and text content"""
        target_normalized = NameMatcher.normalize_name(target_name)
        text_normalized = NameMatcher.normalize_name(text)

        # Check for exact match
        if target_normalized in text_normalized:
            return 1.0

        # Check variations
        variations = NameMatcher.get_name_variations(target_name)
        max_similarity = 0.0

        for variation in variations:
            if variation in text_normalized:
                similarity = len(variation) / len(target_normalized)
                max_similarity = max(max_similarity, similarity)

        # Use sequence matcher for fuzzy matching
        target_parts = target_normalized.split()
        for part in target_parts:
            if len(part) > 2:  # Only check meaningful parts
                matcher = SequenceMatcher(None, part, text_normalized)
                similarity = matcher.ratio()
                if similarity > 0.7:  # High similarity threshold
                    max_similarity = max(max_similarity, similarity * 0.8)  # Weight down fuzzy matches

        return max_similarity

    @staticmethod
    def is_name_relevant(target_name: str, content: str, metadata: Dict[str, str], url: str) -> bool:
        """Enhanced relevance checking with name variations"""
        # Combine all text for checking
        all_text = f"{metadata.get('title', '')} {metadata.get('description', '')} {content} {url}".lower()

        # Calculate similarity score
        similarity_score = NameMatcher.calculate_name_similarity(target_name, all_text)

        # Log the similarity for debugging
        logger.info(f"Name similarity score for {target_name} on {url}: {similarity_score:.2f}")

        # Accept if similarity is above threshold
        return similarity_score > 0.5  # Increased threshold for better precision


# OPTIMIZED: Enhanced web scraper with no retries
class OptimizedWebScraper:
    """Optimized web scraper with single attempt and fast timeouts"""

    def __init__(self):
        self.session = None
        if ENHANCED_SCRAPING:
            self.scraper = cloudscraper.create_scraper(
                browser={
                    'browser': 'chrome',
                    'platform': 'windows',
                    'mobile': False
                }
            )
        else:
            self.scraper = None

    async def __aenter__(self):
        connector = aiohttp.TCPConnector(
            limit=50,  # OPTIMIZED: Increased connection pool
            limit_per_host=10,  # OPTIMIZED: More connections per host
            ttl_dns_cache=300,
            use_dns_cache=True,
        )

        timeout = ClientTimeout(total=URL_FETCH_TIMEOUT, connect=10, sock_read=20)

        self.session = ClientSession(
            connector=connector,
            timeout=timeout,
            headers=self._get_headers()
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    def _get_headers(self):
        """Get headers with user agent"""
        if ENHANCED_SCRAPING:
            user_agent = ua.random
        else:
            user_agent = random.choice(USER_AGENTS)

        return {
            'User-Agent': user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Cache-Control': 'max-age=0',
            'Pragma': 'no-cache',
        }

    async def fetch_url(self, url: str) -> Optional[str]:
        """OPTIMIZED: Fetch URL with single attempt only"""
        try:
            # Skip PDF files and other non-HTML content
            if url.lower().endswith(('.pdf', '.doc', '.docx', '.xls', '.xlsx')):
                logger.warning(f"Skipping non-HTML file: {url}")
                return None

            # Special handling for LinkedIn (known to block)
            if 'linkedin.com' in url.lower():
                logger.warning(f"LinkedIn URL detected: {url}")
                if ENHANCED_SCRAPING:
                    return self._fetch_with_cloudscraper(url)
                else:
                    logger.warning(f"LinkedIn blocks requests, skipping: {url}")
                    return None

            headers = self._get_headers()

            async with self.session.get(url, headers=headers, allow_redirects=True) as response:
                if response.status == 403 or response.status == 429 or response.status == 999:
                    # Try with cloudscraper for protected sites
                    if ENHANCED_SCRAPING:
                        logger.info(f"Blocked access ({response.status}), trying cloudscraper: {url}")
                        return self._fetch_with_cloudscraper(url)
                    else:
                        logger.warning(f"Blocked access ({response.status}): {url}")
                        return None

                if response.status >= 400:
                    logger.warning(f"HTTP {response.status} for {url}")
                    return None

                content_type = response.headers.get('Content-Type', '').lower()
                if not any(ct in content_type for ct in ['text/html', 'application/xhtml+xml']):
                    logger.warning(f"Unsupported content type {content_type} for {url}")
                    return None

                try:
                    html_content = await response.text()
                    if len(html_content) < 500:  # Minimum content threshold
                        logger.warning(f"Content too small for {url}")
                        return None

                    return html_content
                except UnicodeDecodeError:
                    # Try with different encoding
                    html_content = await response.read()
                    return html_content.decode('utf-8', errors='ignore')

        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
            return None

    def _fetch_with_cloudscraper(self, url: str) -> Optional[str]:
        """Fallback method using cloudscraper for protected sites"""
        if not ENHANCED_SCRAPING:
            return None

        try:
            response = self.scraper.get(url, timeout=30)
            if response.status_code == 200:
                return response.text
            else:
                logger.warning(f"Cloudscraper failed with status {response.status_code} for {url}")
                return None
        except Exception as e:
            logger.error(f"Cloudscraper failed for {url}: {e}")
            return None


class GoogleCustomSearchAPI:
    """Google Custom Search API implementation"""

    def __init__(self):
        self.api_key = os.getenv("GOOGLE_CUSTOM_SEARCH_API_KEY")
        self.search_engine_id = os.getenv("GOOGLE_CUSTOM_SEARCH_ENGINE_ID")
        self.base_url = "https://www.googleapis.com/customsearch/v1"

    def is_available(self) -> bool:
        """Check if Google Custom Search API is properly configured"""
        return bool(self.api_key and self.search_engine_id)

    async def search(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """
        Search using Google Custom Search API

        Args:
            query: Search query string
            max_results: Maximum number of results to return (max 100 per day for free tier)

        Returns:
            List of SearchResult objects
        """
        if not self.is_available():
            logger.warning("Google Custom Search API not configured")
            return []

        try:
            results = []
            # Google Custom Search API returns max 10 results per request
            requests_needed = min((max_results + 9) // 10, 10)  # Max 10 requests = 100 results

            for page in range(requests_needed):
                start_index = page * 10 + 1
                page_results = await self._search_page(query, start_index)
                results.extend(page_results)

                # Stop if we have enough results or no more results
                if len(results) >= max_results or len(page_results) < 10:
                    break

                # Add delay between requests to respect rate limits
                if page < requests_needed - 1:
                    await asyncio.sleep(0.1)

            # Limit to requested number of results
            return results[:max_results]

        except Exception as e:
            logger.error(f"Google Custom Search API error: {e}")
            return []

    async def _search_page(self, query: str, start_index: int = 1) -> List[SearchResult]:
        """Search a single page using Google Custom Search API"""
        params = {
            'key': self.api_key,
            'cx': self.search_engine_id,
            'q': query,
            'start': start_index,
            'num': 10,  # Max results per request
            'safe': 'medium',
            'fields': 'items(title,link,snippet,displayLink),searchInformation(totalResults)'
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_custom_search_results(data, query)
                    elif response.status == 429:
                        logger.warning("Google Custom Search API rate limit exceeded")
                        return []
                    elif response.status == 403:
                        logger.warning("Google Custom Search API quota exceeded or invalid credentials")
                        return []
                    else:
                        logger.warning(f"Google Custom Search API returned status {response.status}")
                        return []

        except asyncio.TimeoutError:
            logger.error("Google Custom Search API timeout")
            return []
        except Exception as e:
            logger.error(f"Error in Google Custom Search API request: {e}")
            return []

    def _parse_custom_search_results(self, data: dict, query: str) -> List[SearchResult]:
        """Parse Google Custom Search API response"""
        results = []

        items = data.get('items', [])

        for item in items:
            try:
                url = item.get('link', '')
                title = item.get('title', '')
                snippet = item.get('snippet', '')
                display_link = item.get('displayLink', '')

                # Skip invalid URLs
                if not self._is_valid_url(url):
                    continue

                # Calculate relevance score
                relevance_score = self._calculate_relevance(query, title, snippet, url)

                # Create SearchResult
                search_result = SearchResult(
                    url=url,
                    title=title,
                    description=snippet,
                    domain=display_link or urlparse(url).netloc,
                    relevance_score=relevance_score,
                    content_type="webpage"
                )

                results.append(search_result)
                logger.debug(f"Google Custom Search result: {title} - {url}")

            except Exception as e:
                logger.warning(f"Error parsing Google Custom Search result: {e}")
                continue

        logger.info(f"Google Custom Search returned {len(results)} results")
        return results

    def _is_valid_url(self, url: str) -> bool:
        """Validate if URL is worth including"""
        if not url:
            return False

        # Skip invalid URL schemes
        invalid_schemes = ['javascript:', 'mailto:', 'tel:', 'data:', '#']
        if any(url.lower().startswith(scheme) for scheme in invalid_schemes):
            return False

        # Must start with http or https
        if not url.startswith(('http://', 'https://')):
            return False

        # Skip unwanted domains
        unwanted_domains = [
            'google.com/search', 'google.com/url', 'webcache.googleusercontent.com',
            'accounts.google.com', 'support.google.com', 'policies.google.com',
            'translate.google.com', 'maps.google.com'
        ]

        if any(domain in url.lower() for domain in unwanted_domains):
            return False

        return True

    def _calculate_relevance(self, query: str, title: str, description: str, url: str = "") -> int:
        """Calculate relevance score for Custom Search results"""
        score = 0
        query_terms = query.lower().split()

        title_lower = title.lower() if title else ""
        description_lower = description.lower() if description else ""
        url_lower = url.lower() if url else ""

        # Score based on individual term matches
        for term in query_terms:
            if term in title_lower:
                score += 15  # High weight for title matches
            if term in description_lower:
                score += 5  # Medium weight for description matches
            if term in url_lower:
                score += 3  # Lower weight for URL matches

        # Bonus for exact query match
        if query.lower() in title_lower:
            score += 25
        if query.lower() in description_lower:
            score += 10

        # Bonus for authoritative domains
        if url:
            high_authority_domains = [
                'wikipedia.org', 'linkedin.com', 'facebook.com', 'twitter.com',
                'instagram.com', 'youtube.com', 'github.com', 'stackoverflow.com',
                'medium.com', 'blogspot.com', 'wordpress.com'
            ]

            domain = urlparse(url).netloc.lower()
            if any(auth_domain in domain for auth_domain in high_authority_domains):
                score += 10

        # Bonus for name-like patterns (capitalized words)
        if any(term.istitle() for term in query.split()):
            score += 5

        return max(score, 1)  # Minimum score of 1


class AdvancedSearchEngine:
    """Enhanced search engine with Google Custom Search API integration"""

    def __init__(self):
        self.serpapi_key = os.getenv("SERPAPI_KEY")
        self.session = None
        self.google_custom_search = GoogleCustomSearchAPI()

        # Enhanced user agents for better success rate
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        ]

    async def __aenter__(self):
        connector = aiohttp.TCPConnector(
            limit=20,
            limit_per_host=5,
            ttl_dns_cache=300,
            use_dns_cache=True
        )
        timeout = ClientTimeout(total=30, connect=10, sock_read=20)
        self.session = ClientSession(
            connector=connector,
            timeout=timeout,
            headers=self._get_headers()
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    def _get_headers(self):
        """Get realistic headers to avoid bot detection"""
        return {
            'User-Agent': random.choice(self.user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Cache-Control': 'max-age=0',
        }

    async def search_multiple_engines(self, query: str, max_results: int = 25) -> List[SearchResult]:
        """Enhanced search using multiple engines with Google Custom Search API as primary"""
        all_results = []

        # Primary: Google Custom Search API (most reliable)
        if self.google_custom_search.is_available():
            try:
                custom_search_results = await self.google_custom_search.search(query, max_results)
                all_results.extend(custom_search_results)
                logger.info(f"Google Custom Search API returned {len(custom_search_results)} results")
            except Exception as e:
                logger.error(f"Google Custom Search API failed: {e}")

        # Fallback 1: Direct Google search with BeautifulSoup (if we need more results)
        if len(all_results) < max_results:
            try:
                remaining_results = max_results - len(all_results)
                google_results = await self._search_google_direct(query, remaining_results)
                all_results.extend(google_results)
                logger.info(f"Direct Google search returned {len(google_results)} additional results")
            except Exception as e:
                logger.error(f"Direct Google search failed: {e}")

        # Fallback 2: SerpAPI if available and we need more results
        if len(all_results) < max_results and self.serpapi_key and SERPAPI_AVAILABLE:
            try:
                remaining_results = max_results - len(all_results)
                serpapi_results = await self._search_serpapi(query, remaining_results)
                all_results.extend(serpapi_results)
                logger.info(f"SerpAPI returned {len(serpapi_results)} additional results")
            except Exception as e:
                logger.error(f"SerpAPI search failed: {e}")

        # Fallback 3: Google search library if available and we still need more results
        if len(all_results) < max_results and GOOGLE_SEARCH_AVAILABLE:
            try:
                remaining_results = max_results - len(all_results)
                google_lib_results = await self._search_google_fallback(query, remaining_results)
                all_results.extend(google_lib_results)
                logger.info(f"Google library search returned {len(google_lib_results)} additional results")
            except Exception as e:
                logger.error(f"Google library search failed: {e}")

        # Fallback 4: Bing search
        if len(all_results) < max_results:
            try:
                remaining_results = max_results - len(all_results)
                bing_results = await self._search_bing(query, remaining_results)
                all_results.extend(bing_results)
                logger.info(f"Bing search returned {len(bing_results)} additional results")
            except Exception as e:
                logger.error(f"Bing search failed: {e}")

        # Remove duplicates and sort by relevance
        unique_results = self._deduplicate_results(all_results)
        return sorted(unique_results, key=lambda x: x.relevance_score, reverse=True)[:max_results]

    async def _search_google_direct(self, query: str, max_results: int) -> List[SearchResult]:
        """
        Direct Google search using BeautifulSoup - Fallback method
        """
        if not self.session:
            return []

        try:
            # Construct Google search URL
            encoded_query = quote_plus(query)
            search_url = f"https://www.google.com/search?q={encoded_query}&num={min(max_results, 100)}&hl=en"

            logger.info(f"Fetching Google search results for: {query}")

            # Add random delay to avoid rate limiting
            await asyncio.sleep(random.uniform(1.0, 10.0))

            # Fetch the search results page
            headers = self._get_headers()
            async with self.session.get(search_url, headers=headers, allow_redirects=True) as response:
                if response.status != 200:
                    logger.warning(f"Google search returned status {response.status}")
                    return []

                html_content = await response.text()

                # Parse with BeautifulSoup
                soup = BeautifulSoup(html_content, 'html.parser')

                # Extract search results
                search_results = self._parse_google_results(soup, query)

                logger.info(f"Successfully parsed {len(search_results)} results from Google")
                return search_results

        except Exception as e:
            logger.error(f"Error in direct Google search: {e}")
            return []

    def _parse_google_results(self, soup: BeautifulSoup, query: str) -> List[SearchResult]:
        """
        Parse Google search results from HTML using BeautifulSoup
        """
        results = []

        # Multiple selectors for Google result containers (Google changes these frequently)
        result_selectors = [
            'div.g',  # Main result container
            'div.tF2Cxc',  # Common container class
            'div[data-ved]',  # Container with data-ved attribute
            '.rc',  # Classic result container
            'div.yuRUbf',  # Another common container
        ]

        result_containers = []
        for selector in result_selectors:
            containers = soup.select(selector)
            if containers:
                result_containers = containers
                logger.info(f"Found {len(containers)} Google results using selector: {selector}")
                break

        if not result_containers:
            logger.warning("No Google result containers found with any selector")
            # Try to find any div that might contain results
            fallback_containers = soup.select('div:has(a[href*="http"])')
            result_containers = fallback_containers[:20]  # Limit fallback results
            logger.info(f"Using fallback: found {len(result_containers)} potential containers")

        for container in result_containers:
            try:
                # Extract URL
                url = self._extract_google_url(container)
                if not url or not self._is_valid_url(url):
                    continue

                # Extract title
                title = self._extract_google_title(container)
                if not title:
                    continue

                # Extract description
                description = self._extract_google_description(container)

                # Create domain from URL
                try:
                    domain = urlparse(url).netloc
                except:
                    domain = url.split('/')[2] if '/' in url else url

                # Calculate relevance score
                relevance_score = self._calculate_relevance(query, title, description, url)

                # Create SearchResult object
                search_result = SearchResult(
                    url=url,
                    title=title,
                    description=description,
                    domain=domain,
                    relevance_score=relevance_score,
                    content_type="webpage"
                )

                results.append(search_result)
                logger.debug(f"Extracted Google result: {title} - {url}")

            except Exception as e:
                logger.warning(f"Error parsing individual Google result: {e}")
                continue

        return results

    def _extract_google_url(self, container) -> Optional[str]:
        """Extract URL from Google result container"""
        # Multiple strategies for URL extraction
        url_strategies = [
            # Strategy 1: Most common - h3 within a link
            lambda c: c.select_one('h3 a'),
            # Strategy 2: Any link in the yuRUbf container
            lambda c: c.select_one('.yuRUbf a'),
            # Strategy 3: First link with href
            lambda c: c.select_one('a[href]'),
            # Strategy 4: Link in title area
            lambda c: c.select_one('h2 a, h3 a'),
        ]

        for strategy in url_strategies:
            try:
                link_elem = strategy(container)
                if link_elem and link_elem.get('href'):
                    url = link_elem['href']

                    # Clean up Google redirect URLs
                    if url.startswith('/url?'):
                        # Extract actual URL from Google redirect
                        try:
                            from urllib.parse import parse_qs, urlparse
                            parsed = urlparse(url)
                            url_param = parse_qs(parsed.query).get('url', [])
                            if url_param:
                                url = url_param[0]
                            else:
                                # Try 'q' parameter as fallback
                                q_param = parse_qs(parsed.query).get('q', [])
                                if q_param:
                                    url = q_param[0]
                        except:
                            continue

                    # Skip Google internal links
                    google_internal = [
                        'google.com/search', 'webcache.googleusercontent.com',
                        'accounts.google.com', 'support.google.com',
                        'policies.google.com', 'translate.google.com'
                    ]

                    if any(internal in url for internal in google_internal):
                        continue

                    return url
            except:
                continue

        return None

    def _extract_google_title(self, container) -> Optional[str]:
        """Extract title from Google result container"""
        title_strategies = [
            # Strategy 1: h3 tag (most common)
            lambda c: c.select_one('h3'),
            # Strategy 2: h2 tag (alternative)
            lambda c: c.select_one('h2'),
            # Strategy 3: Link text within h3
            lambda c: c.select_one('h3 a'),
            # Strategy 4: Specific Google classes
            lambda c: c.select_one('.LC20lb, .DKV0Md'),
            # Strategy 5: Any heading with role
            lambda c: c.select_one('[role="heading"]'),
        ]

        for strategy in title_strategies:
            try:
                title_elem = strategy(container)
                if title_elem:
                    title = title_elem.get_text().strip()
                    if title and len(title) > 1:
                        return title
            except:
                continue

        return None

    def _extract_google_description(self, container) -> str:
        """Extract description/snippet from Google result container"""
        description_strategies = [
            # Strategy 1: Common snippet classes
            lambda c: c.select_one('.VwiC3b, .s3v9rd, .IsZvec'),
            # Strategy 2: Classic snippet class
            lambda c: c.select_one('.st'),
            # Strategy 3: Styled span elements
            lambda c: c.select_one('span[style*="color"]'),
            # Strategy 4: Data snippet
            lambda c: c.select_one('[data-content-feature="1"]'),
            # Strategy 5: Any span that looks like a snippet
            lambda c: c.select_one('span:not([class*="date"]):not([class*="url"])'),
        ]

        description = ""

        for strategy in description_strategies:
            try:
                desc_elem = strategy(container)
                if desc_elem:
                    desc_text = desc_elem.get_text().strip()
                    if desc_text and len(desc_text) > 20:
                        description = desc_text
                        break
            except:
                continue

        # Fallback: extract meaningful text from container
        if not description:
            try:
                # Get all text and filter for description-like content
                all_text = container.get_text()
                lines = [line.strip() for line in all_text.split('\n') if line.strip()]
                for line in lines:
                    # Look for lines that seem like descriptions
                    if (20 < len(line) < 400 and
                            not line.startswith('http') and
                            not line.endswith('.com') and
                            ' ' in line):  # Has spaces (not just a single word)
                        description = line
                        break
            except:
                pass

        return description

    async def _search_serpapi(self, query: str, max_results: int) -> List[SearchResult]:
        """Search using SerpAPI - keeping existing implementation"""
        if not SERPAPI_AVAILABLE or not self.serpapi_key:
            return []

        params = {
            "engine": "google",
            "q": query,
            "api_key": self.serpapi_key,
            "num": min(max_results, 100),
            "safe": "active"
        }

        try:
            search_instance = serpapi.GoogleSearch(params)
            results = search_instance.get_dict()

            search_results = []
            for result in results.get("organic_results", []):
                search_results.append(SearchResult(
                    url=result.get("link", ""),
                    title=result.get("title", ""),
                    description=result.get("snippet", ""),
                    domain=urlparse(result.get("link", "")).netloc,
                    relevance_score=self._calculate_relevance(query, result.get("title", ""), result.get("snippet", ""))
                ))

            return search_results
        except Exception as e:
            logger.error(f"SerpAPI error: {e}")
            return []

    async def _search_google_fallback(self, query: str, max_results: int) -> List[SearchResult]:
        """Fallback Google search using googlesearch library"""
        if not GOOGLE_SEARCH_AVAILABLE:
            return []

        try:
            def sync_google_search():
                urls = []
                try:
                    for url in google_search(query, tld="com", num=max_results, stop=max_results, pause=2):
                        urls.append(url)
                except Exception as e:
                    logger.error(f"Google search library error: {e}")
                return urls

            urls = await asyncio.to_thread(sync_google_search)

            search_results = []
            for url in urls:
                parsed_url = urlparse(url)
                search_results.append(SearchResult(
                    url=url,
                    title=f"Search result from {parsed_url.netloc}",
                    description="",
                    domain=parsed_url.netloc,
                    relevance_score=self._calculate_basic_relevance(query, url)
                ))

            return search_results

        except Exception as e:
            logger.error(f"Google fallback search error: {e}")
            return []

    async def _search_bing(self, query: str, max_results: int) -> List[SearchResult]:
        """Enhanced Bing search implementation"""
        if not self.session:
            return []

        try:
            headers = self._get_headers()
            search_url = f"https://www.bing.com/search?q={quote_plus(query)}&count={max_results}"

            async with self.session.get(search_url, headers=headers) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    return self._parse_bing_results(soup, query)
                else:
                    logger.warning(f"Bing search failed with status {response.status}")
                    return []

        except Exception as e:
            logger.error(f"Bing search error: {e}")
            return []

    def _parse_bing_results(self, soup: BeautifulSoup, query: str) -> List[SearchResult]:
        """Parse Bing search results from HTML - keeping existing implementation but improved"""
        try:
            results = []

            # Look for search result containers
            result_containers = soup.find_all('li', class_='b_algo') + soup.find_all('div', class_='b_algo')

            for container in result_containers[:15]:  # Limit to first 15 results
                try:
                    title_elem = container.find('h2') or container.find('h3')
                    if not title_elem:
                        continue

                    link_elem = title_elem.find('a')
                    if not link_elem or not link_elem.get('href'):
                        continue

                    url = link_elem['href']
                    title = link_elem.get_text().strip()

                    # Get description
                    desc_elem = container.find('p') or container.find('div', class_='b_caption')
                    description = desc_elem.get_text().strip() if desc_elem else ""

                    # Skip if URL is invalid
                    if not self._is_valid_url(url):
                        continue

                    results.append(SearchResult(
                        url=url,
                        title=title,
                        description=description,
                        domain=urlparse(url).netloc,
                        relevance_score=self._calculate_relevance(query, title, description)
                    ))

                except Exception as e:
                    logger.warning(f"Error parsing Bing result: {e}")
                    continue

            logger.info(f"Bing search parsed {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Error parsing Bing results: {e}")
            return []

    def _is_valid_url(self, url: str) -> bool:
        """Validate if URL is worth including"""
        if not url:
            return False

        # Skip invalid URL schemes
        invalid_schemes = ['javascript:', 'mailto:', 'tel:', 'data:', '#']
        if any(url.lower().startswith(scheme) for scheme in invalid_schemes):
            return False

        # Must start with http or https
        if not url.startswith(('http://', 'https://')):
            return False

        # Skip unwanted domains
        unwanted_domains = [
            'google.com/search', 'google.com/url', 'webcache.googleusercontent.com',
            'accounts.google.com', 'support.google.com', 'policies.google.com',
            'translate.google.com', 'maps.google.com'
        ]

        if any(domain in url.lower() for domain in unwanted_domains):
            return False

        return True

    def _calculate_relevance(self, query: str, title: str, description: str, url: str = "") -> int:
        """Enhanced relevance calculation"""
        score = 0
        query_terms = query.lower().split()

        title_lower = title.lower() if title else ""
        description_lower = description.lower() if description else ""
        url_lower = url.lower() if url else ""

        # Score based on individual term matches
        for term in query_terms:
            if term in title_lower:
                score += 15  # High weight for title matches
            if term in description_lower:
                score += 5  # Medium weight for description matches
            if term in url_lower:
                score += 3  # Lower weight for URL matches

        # Bonus for exact query match
        if query.lower() in title_lower:
            score += 25
        if query.lower() in description_lower:
            score += 10

        # Bonus for authoritative domains
        if url:
            high_authority_domains = [
                'wikipedia.org', 'linkedin.com', 'facebook.com', 'twitter.com',
                'instagram.com', 'youtube.com', 'github.com', 'stackoverflow.com',
                'medium.com', 'blogspot.com', 'wordpress.com'
            ]

            domain = urlparse(url).netloc.lower()
            if any(auth_domain in domain for auth_domain in high_authority_domains):
                score += 10

        # Bonus for name-like patterns (capitalized words)
        if any(term.istitle() for term in query.split()):
            score += 5

        return max(score, 1)  # Minimum score of 1

    def _calculate_basic_relevance(self, query: str, url: str) -> int:
        """Basic relevance for URLs without description"""
        score = 1
        query_lower = query.lower()
        url_lower = url.lower()

        if query_lower in url_lower:
            score += 8

        for term in query_lower.split():
            if term in url_lower:
                score += 3

        return score

    def _deduplicate_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """Remove duplicate URLs and normalize domains"""
        seen_urls = set()
        unique_results = []

        for result in results:
            # Normalize URL for comparison
            normalized_url = result.url.lower().rstrip('/')

            # Also check for very similar URLs (with/without www, trailing slashes, etc.)
            base_url = normalized_url.replace('www.', '').rstrip('/')

            if base_url not in seen_urls:
                seen_urls.add(base_url)
                unique_results.append(result)

        return unique_results


class SocialMediaSearcher:
    """Enhanced social media searcher with better platform detection"""

    def __init__(self, search_engine: AdvancedSearchEngine):
        self.search_engine = search_engine

    async def find_social_accounts(self, name: str) -> Dict[str, List[SearchResult]]:
        """Find social media accounts with enhanced platform-specific searching"""
        platforms = {
            "facebook": [f'"{name}" site:facebook.com'],
            "twitter": [f'"{name}" site:twitter.com', f'"{name}" site:x.com'],
            "instagram": [f'"{name}" site:instagram.com'],
            "linkedin": [f'"{name}" site:linkedin.com/in/', f'"{name}" site:linkedin.com/pub/'],
            "tiktok": [f'"{name}" site:tiktok.com'],
            "youtube": [f'"{name}" site:youtube.com/channel/', f'"{name}" site:youtube.com/c/',
                        f'"{name}" site:youtube.com/@'],
        }

        social_results = {}

        for platform, queries in platforms.items():
            try:
                platform_results = []

                # Try each query for the platform
                for query in queries:
                    results = await self.search_engine.search_multiple_engines(query, max_results=5)

                    # Filter and validate social media URLs
                    for result in results:
                        if self._is_valid_social_url(result.url, platform):
                            result.platform = platform
                            result.username = self._extract_username(result.url, platform)
                            result.content_type = "social_media"
                            platform_results.append(result)

                # Remove duplicates within platform
                platform_results = self._deduplicate_social_results(platform_results)
                social_results[platform] = platform_results[:3]  # Limit to top 3 per platform
                logger.info(f"Found {len(platform_results)} {platform} profiles")

                # Add delay between platform searches
                await asyncio.sleep(1)

            except Exception as e:
                logger.error(f"Error searching {platform} for {name}: {e}")
                social_results[platform] = []

        return social_results

    def _is_valid_social_url(self, url: str, platform: str) -> bool:
        """Enhanced validation for social media URLs"""
        url_lower = url.lower()

        # Common exclusions for all platforms
        common_exclusions = ["help", "support", "about", "privacy", "terms", "login", "signup", "posts", "photos",
                             "videos"]
        if any(exclusion in url_lower for exclusion in common_exclusions):
            return False

        # Platform-specific validation
        if platform == "facebook":
            return ("facebook.com/" in url_lower and
                    not any(exclude in url_lower for exclude in
                            ["events", "groups", "pages/category", "marketplace", "gaming", "watch"]))

        elif platform == "twitter":
            return (("twitter.com/" in url_lower or "x.com/" in url_lower) and
                    not any(exclude in url_lower for exclude in
                            ["status", "search", "hashtag", "lists", "i/web", "explore", "home"]))

        elif platform == "instagram":
            return ("instagram.com/" in url_lower and
                    not any(exclude in url_lower for exclude in
                            ["p/", "explore", "tv", "reel", "stories", "direct"]))

        elif platform == "linkedin":
            return ("linkedin.com/" in url_lower and
                    any(valid in url_lower for valid in ["/in/", "/pub/"]))

        elif platform == "tiktok":
            return ("tiktok.com/" in url_lower and
                    ("@" in url_lower or "/user/" in url_lower))

        elif platform == "youtube":
            return ("youtube.com/" in url_lower and
                    any(valid in url_lower for valid in ["/c/", "/@", "/channel/", "/user/"]))

        return False

    def _extract_username(self, url: str, platform: str) -> str:
        """Enhanced username extraction"""
        try:
            parsed = urlparse(url)
            path = parsed.path.strip('/')

            if platform == "facebook":
                if 'profile.php' in url:
                    return parse_qs(parsed.query).get('id', [''])[0]
                if 'people/' in path:
                    return path.split('people/')[1].split('/')[0]
                return path.split('/')[0] if path else ''

            elif platform == "twitter":
                return path.split('/')[0] if path else ''

            elif platform == "instagram":
                return path.split('/')[0] if path else ''

            elif platform == "linkedin":
                if '/in/' in path:
                    return path.split('/in/')[1].split('/')[0]
                elif '/pub/' in path:
                    return path.split('/pub/')[1].split('/')[0]
                return ''

            elif platform == "tiktok":
                if '@' in path:
                    return path.replace('@', '').split('/')[0]
                elif '/user/' in path:
                    return path.split('/user/')[1].split('/')[0]
                return path.split('/')[0] if path else ''

            elif platform == "youtube":
                for prefix in ['/c/', '/@', '/channel/', '/user/']:
                    if prefix in path:
                        return path.split(prefix)[1].split('/')[0]
                return ''

            return ''

        except Exception as e:
            logger.error(f"Error extracting username from {url}: {e}")
            return ''

    def _deduplicate_social_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """Remove duplicate social media results"""
        seen_urls = set()
        unique_results = []

        for result in results:
            normalized_url = result.url.lower().rstrip('/')
            if normalized_url not in seen_urls:
                seen_urls.add(normalized_url)
                unique_results.append(result)

        return unique_results


class ContentProcessor:
    """Enhanced content processor with improved extraction and relevance filtering"""

    def __init__(self, openai_client):
        self.client = openai_client

    async def process_webpage(self, html_content: str, url: str, target_name: str) -> ExtractionResult:
        """Process webpage content with enhanced extraction"""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')

            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'iframe']):
                element.decompose()

            # Remove comments
            for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
                comment.extract()

            # Extract metadata
            metadata = self._extract_metadata(soup, url)

            # Extract main content
            main_content = self._extract_main_content(soup)

            # Use enhanced name matching for relevance
            if not NameMatcher.is_name_relevant(target_name, main_content, metadata, url):
                logger.info(f"Content not relevant for {target_name} on {url}")
                return ExtractionResult(
                    source=url,
                    content=[],
                    metadata=metadata,
                    success=True
                )

            logger.info(f"Content appears relevant for {target_name} on {url}, processing with GPT")

            # Process with GPT
            processed_content = await self._process_with_gpt(
                main_content, metadata, url, target_name, "webpage"
            )

            return ExtractionResult(
                source=url,
                content=processed_content,
                metadata=metadata,
                success=True
            )

        except Exception as e:
            logger.error(f"Error processing webpage {url}: {e}")
            return ExtractionResult(
                source=url,
                content=[],
                metadata={},
                success=False,
                error=str(e)
            )

    async def process_social_media(self, html_content: str, url: str, platform: str, username: str,
                                   target_name: str) -> ExtractionResult:
        """Enhanced social media content processing"""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            metadata = self._extract_social_metadata(soup, url, platform)

            # For social media, always try to extract since URLs often contain names
            logger.info(f"Processing {platform} profile for {target_name}")

            # Extract any visible text content for social media
            visible_content = self._extract_social_content(soup, platform)

            # Process with GPT
            processed_content = await self._process_with_gpt(
                visible_content, metadata, url, target_name, "social_media", platform, username
            )

            return ExtractionResult(
                source=url,
                content=processed_content,
                metadata=metadata,
                success=True
            )

        except Exception as e:
            logger.error(f"Error processing social media {url}: {e}")
            return ExtractionResult(
                source=url,
                content=[],
                metadata={},
                success=False,
                error=str(e)
            )

    def _extract_metadata(self, soup: BeautifulSoup, url: str) -> Dict[str, str]:
        """Comprehensive metadata extraction"""
        metadata = {
            'url': url,
            'domain': urlparse(url).netloc,
            'title': '',
            'description': '',
            'keywords': '',
            'author': '',
            'published_date': '',
            'og_title': '',
            'og_description': '',
            'og_site_name': '',
            'twitter_title': '',
            'twitter_description': '',
        }

        # Basic metadata
        if soup.title:
            metadata['title'] = soup.title.string.strip()

        # Meta tags
        meta_tags = {
            'description': ['name', 'description'],
            'keywords': ['name', 'keywords'],
            'author': ['name', 'author'],
            'published_date': ['name', 'date'],
            'og_title': ['property', 'og:title'],
            'og_description': ['property', 'og:description'],
            'og_site_name': ['property', 'og:site_name'],
            'twitter_title': ['name', 'twitter:title'],
            'twitter_description': ['name', 'twitter:description'],
        }

        for key, (attr, value) in meta_tags.items():
            tag = soup.find('meta', {attr: value})
            if tag and tag.get('content'):
                metadata[key] = tag['content'].strip()

        return metadata

    def _extract_social_metadata(self, soup: BeautifulSoup, url: str, platform: str) -> Dict[str, str]:
        """Extract social media specific metadata"""
        metadata = self._extract_metadata(soup, url)
        metadata['platform'] = platform

        # Extract username from URL
        username = extract_username_from_url(url)
        if username:
            metadata['username'] = username

        # Platform-specific extraction
        if platform == 'linkedin':
            # LinkedIn often has job titles in the title
            title = metadata.get('title', '')
            if ' - ' in title:
                parts = title.split(' - ')
                if len(parts) > 1:
                    metadata['job_title'] = parts[1]

        elif platform == 'facebook':
            # Extract Facebook-specific metadata
            fb_title = soup.find('title')
            if fb_title:
                metadata['profile_name'] = fb_title.get_text().split(' | ')[0]

        elif platform == 'twitter':
            # Extract Twitter-specific metadata
            twitter_title = metadata.get('title', '')
            if ' (@' in twitter_title:
                parts = twitter_title.split(' (@')
                if len(parts) > 1:
                    metadata['display_name'] = parts[0]
                    metadata['handle'] = '@' + parts[1].split(')')[0]

        return metadata

    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """Extract main content from webpage with better content selection"""
        # Try to find main content areas in order of preference
        content_selectors = [
            'main', 'article', '[role="main"]',
            '.content', '#content', '.main-content',
            '.post-content', '.entry-content', '.page-content',
            '.article-content', '.blog-content'
        ]

        main_content = ""

        for selector in content_selectors:
            elements = soup.select(selector)
            if elements:
                main_content = ' '.join([elem.get_text() for elem in elements])
                break

        # Fallback to body content if no main content found
        if not main_content:
            body = soup.find('body')
            if body:
                main_content = body.get_text()

        # Clean up the text
        lines = main_content.split('\n')
        cleaned_lines = [line.strip() for line in lines if line.strip() and len(line.strip()) > 3]

        return ' '.join(cleaned_lines)[:15000]  # Limit content size

    def _extract_social_content(self, soup: BeautifulSoup, platform: str) -> str:
        """Extract visible content from social media pages"""
        content = ""

        # Platform-specific content extraction
        if platform == 'linkedin':
            # LinkedIn profile content
            selectors = ['.pv-about-section', '.pv-top-card', '.experience-section', '.education-section']
        elif platform == 'facebook':
            # Facebook profile content
            selectors = ['[data-overviewsection]', '.userContentWrapper', '.fbTimelineSection']
        elif platform == 'twitter':
            # Twitter profile content
            selectors = ['[data-testid="UserDescription"]', '.ProfileHeaderCard', '.tweet']
        elif platform == 'instagram':
            # Instagram profile content
            selectors = ['.-vDIg', '.C4VMK', 'article']
        else:
            # Generic selectors
            selectors = ['main', 'article', '.content', '.profile', '.bio']

        for selector in selectors:
            elements = soup.select(selector)
            if elements:
                content += ' '.join([elem.get_text() for elem in elements])

        # Fallback to any visible text
        if not content:
            content = soup.get_text()

        # Clean up
        lines = content.split('\n')
        cleaned_lines = [line.strip() for line in lines if line.strip()]
        return ' '.join(cleaned_lines)[:5000]  # Smaller limit for social media

    async def _process_with_gpt(self, content: str, metadata: Dict[str, str], url: str,
                                target_name: str, content_type: str, platform: str = "",
                                username: str = "") -> List[str]:
        """Enhanced GPT processing with better prompts"""

        if content_type == "social_media":
            system_prompt = (
                f"You are an expert at extracting personal information from social media profiles. "
                f"Focus on finding information that is directly related to {target_name}. "
                f"Be very strict about name matching - only extract information if you're confident it's about the target person."
            )

            user_prompt = (
                f"Analyze this {platform} profile to find information about '{target_name}'.\n\n"
                f"URL: {url}\n"
                f"Username: {username}\n"
                f"Title: {metadata.get('title', 'N/A')}\n"
                f"Description: {metadata.get('description', 'N/A')}\n\n"
                f"Content:\n{content[:3000]}\n\n"
                f"IMPORTANT: Only extract information if you're confident this profile belongs to '{target_name}'. "
                f"Look for exact name matches or very close variations. If the profile doesn't clearly belong to "
                f"'{target_name}', return 'NO_RELEVANT_INFORMATION'.\n\n"
                f"If this is the correct person, extract:\n"
                f"- Profile information and bio\n"
                f"- Location data\n"
                f"- Employment/education info\n"
                f"- Contact information\n"
                f"- Any other personal details\n\n"
                f"Format as detailed paragraphs. Be specific about what information you found and where."
            )
        else:
            system_prompt = (
                f"You are an expert at extracting personal information from web content. "
                f"Focus on finding information specifically about {target_name}. "
                f"Be very strict about relevance - only extract information that clearly relates to the target person."
            )

            user_prompt = (
                f"Analyze this webpage to find information about '{target_name}'.\n\n"
                f"URL: {url}\n"
                f"Title: {metadata.get('title', 'N/A')}\n"
                f"Description: {metadata.get('description', 'N/A')}\n\n"
                f"Content:\n{content[:12000]}\n\n"
                f"IMPORTANT: Only extract information if it clearly relates to '{target_name}'. "
                f"Look for exact name matches or very close variations. If the content doesn't clearly relate to "
                f"'{target_name}', return 'NO_RELEVANT_INFORMATION'.\n\n"
                f"If this content is about the target person, extract comprehensive information including:\n"
                f"- Personal details (name, age, location)\n"
                f"- Contact information (email, phone, address)\n"
                f"- Professional information (employer, job title)\n"
                f"- Educational background\n"
                f"- Family information\n"
                f"- Social media profiles mentioned\n"
                f"- Any other personal identifiable information\n\n"
                f"Format as detailed paragraphs with context."
            )

        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=4000
            )

            result = response.choices[0].message.content.strip()

            if result == "NO_RELEVANT_INFORMATION":
                return []

            # Split into paragraphs and clean
            paragraphs = result.split('\n\n')
            cleaned_paragraphs = [p.strip() for p in paragraphs if p.strip() and len(p.strip()) > 20]

            # Log what was extracted
            if cleaned_paragraphs:
                logger.info(f"GPT extracted {len(cleaned_paragraphs)} paragraphs of information from {content_type}")
            else:
                logger.info("GPT found no relevant information")

            return cleaned_paragraphs

        except Exception as e:
            logger.error(f"Error processing with GPT: {e}")
            return []


class PIIExtractor:
    """Enhanced PII extraction with better accuracy and comprehensive attribute mapping"""

    def __init__(self, openai_client):
        self.client = openai_client

    async def extract_pii(self, content_list: List[str], target_name: str) -> Dict[str, Set[str]]:
        """OPTIMIZED: Extract PII from content list with concurrent processing"""

        # Initialize comprehensive PII attributes
        attributes = {
            'Name': set(),
            'Location': set(),
            'Email': set(),
            'Phone': set(),
            'DOB': set(),
            'Address': set(),
            'Gender': set(),
            'Employer': set(),
            'Education': set(),
            'Birth Place': set(),
            'Personal Cell': set(),
            'Business Phone': set(),
            'Facebook Account': set(),
            'Twitter Account': set(),
            'Instagram Account': set(),
            'LinkedIn Account': set(),
            'TikTok Account': set(),
            'YouTube Account': set(),
            'DDL': set(),
            'Passport': set(),
            'Credit Card': set(),
            'SSN': set(),
            'Family Members': set(),
            'Occupation': set(),
            'Salary': set(),
            'Website': set()
        }

        if not content_list:
            return attributes

        # OPTIMIZED: Process content in concurrent batches with rate limiting
        batch_size = 3  # Smaller batches for better concurrency
        semaphore = asyncio.Semaphore(CONCURRENT_GPT_LIMIT)

        async def process_batch_with_semaphore(batch):
            async with semaphore:
                await asyncio.sleep(GPT_RATE_LIMIT_DELAY)  # Rate limiting
                return await self._extract_pii_batch(batch, target_name)

        # Create batches
        batches = [content_list[i:i + batch_size] for i in range(0, len(content_list), batch_size)]

        # Process all batches concurrently
        logger.info(f"Processing {len(batches)} content batches concurrently for PII extraction")
        batch_results = await asyncio.gather(
            *[process_batch_with_semaphore(batch) for batch in batches],
            return_exceptions=True
        )

        # Merge results from all batches
        for batch_result in batch_results:
            if isinstance(batch_result, Exception):
                logger.error(f"Batch processing failed: {batch_result}")
                continue

            for key, values in batch_result.items():
                if key in attributes:
                    attributes[key].update(values)

        return attributes

    async def _extract_pii_batch(self, content_list: List[str], target_name: str) -> Dict[str, Set[str]]:
        """Extract PII from a batch of content"""

        # Initialize attributes
        attributes = {
            'Name': set(), 'Location': set(), 'Email': set(), 'Phone': set(),
            'DOB': set(), 'Address': set(), 'Gender': set(), 'Employer': set(),
            'Education': set(), 'Birth Place': set(), 'Personal Cell': set(),
            'Business Phone': set(), 'Facebook Account': set(), 'Twitter Account': set(),
            'Instagram Account': set(), 'LinkedIn Account': set(), 'TikTok Account': set(),
            'YouTube Account': set(), 'DDL': set(), 'Passport': set(),
            'Credit Card': set(), 'SSN': set(), 'Family Members': set(),
            'Occupation': set(), 'Salary': set(), 'Website': set()
        }

        # Combine content
        input_data = "\n\n".join(content_list)
        logger.info(f"Processing {len(content_list)} content items for PII extraction")

        # Enhanced extraction prompt
        system_prompt = (
            f"You are an advanced PII extraction system. Extract personal information specifically about '{target_name}'. "
            f"Be very precise and only extract information that clearly relates to this specific person. "
            f"Consider slight name variations but be conservative about matches."
        )

        user_prompt = (
            f"Extract all Personally Identifiable Information (PII) about '{target_name}' "
            f"from the following data. Only include information that clearly relates to this specific person.\n\n"
            f"{input_data}\n\n"
            f"Extract these attributes (use empty string if not found):\n"
            f"- Name: Full names, display names, exact name matches\n"
            f"- Location: Current location, cities, states, countries\n"
            f"- Email: Email addresses\n"
            f"- Phone: Phone numbers\n"
            f"- DOB: Date of birth, age, birthday information\n"
            f"- Address: Physical addresses\n"
            f"- Gender: Gender information\n"
            f"- Employer: Current employer, companies\n"
            f"- Education: Schools, degrees, educational background\n"
            f"- Birth Place: Place of birth\n"
            f"- Personal Cell: Mobile phone numbers\n"
            f"- Business Phone: Work phone numbers\n"
            f"- Facebook Account: Facebook profiles/usernames\n"
            f"- Twitter Account: Twitter handles/profiles\n"
            f"- Instagram Account: Instagram usernames/profiles\n"
            f"- LinkedIn Account: LinkedIn profiles\n"
            f"- TikTok Account: TikTok usernames/profiles\n"
            f"- YouTube Account: YouTube channels\n"
            f"- DDL: Driver's license information\n"
            f"- Passport: Passport information\n"
            f"- Credit Card: Credit card information\n"
            f"- SSN: Social Security Numbers\n"
            f"- Family Members: Spouse, children, relatives\n"
            f"- Occupation: Job roles, professional titles\n"
            f"- Salary: Income information\n"
            f"- Website: Personal websites, blogs\n\n"
            f"Return valid JSON with exact field names. For multiple values, use comma separation.\n\n"
            f"Example format:\n"
            f'{{\n'
            f'  "Name": "John Smith",\n'
            f'  "Location": "New York, NY",\n'
            f'  "Email": "john@email.com",\n'
            f'  ...\n'
            f'}}'
        )

        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=4000
            )

            extracted_pii = response.choices[0].message.content.strip()

            # Clean and parse JSON response
            if extracted_pii.startswith('```json'):
                extracted_pii = extracted_pii[7:]
            if extracted_pii.startswith('```'):
                extracted_pii = extracted_pii[3:]
            if extracted_pii.endswith('```'):
                extracted_pii = extracted_pii[:-3]

            extracted_pii = extracted_pii.strip()

            try:
                pii_data = json.loads(extracted_pii)
                logger.info("Successfully parsed JSON response")
            except json.JSONDecodeError:
                try:
                    pii_data = ast.literal_eval(extracted_pii)
                    logger.info("Successfully parsed with ast.literal_eval")
                except:
                    logger.error("Failed to parse PII extraction result")
                    return attributes

            # Update attributes with validation
            updates_made = 0
            for key, value in pii_data.items():
                if key in attributes and value and str(value).strip():
                    value_str = str(value).strip()
                    # Handle comma-separated values
                    if ',' in value_str:
                        for val in value_str.split(','):
                            val = val.strip()
                            if val and self._validate_pii_value(key, val):
                                attributes[key].add(val)
                                updates_made += 1
                    else:
                        if self._validate_pii_value(key, value_str):
                            attributes[key].add(value_str)
                            updates_made += 1

            logger.info(f"PII extraction completed, {updates_made} attributes updated")
            return attributes

        except Exception as e:
            logger.error(f"Error extracting PII: {e}")
            return attributes

    def _validate_pii_value(self, attribute: str, value: str) -> bool:
        """Validate PII values to ensure quality"""
        if not value or len(value.strip()) < 2:
            return False

        # Basic validation for different attribute types
        if attribute == 'Email':
            return '@' in value and '.' in value and len(value) > 5
        elif attribute in ['Phone', 'Personal Cell', 'Business Phone']:
            # Check for phone number patterns
            digits = re.sub(r'[^\d]', '', value)
            return len(digits) >= 7
        elif attribute == 'SSN':
            # Basic SSN format check
            digits = re.sub(r'[^\d]', '', value)
            return len(digits) == 9
        elif attribute in ['Facebook Account', 'Twitter Account', 'Instagram Account', 'LinkedIn Account',
                           'TikTok Account', 'YouTube Account']:
            # Check for valid social media patterns
            return any(platform in value.lower() for platform in
                       ['facebook', 'twitter', 'instagram', 'linkedin', 'tiktok', 'youtube', 'fb.com', 'x.com'])
        elif attribute == 'Website':
            return any(domain in value.lower() for domain in ['http', 'www.', '.com', '.org', '.net'])

        return True


class RiskCalculator:
    """Enhanced risk calculation with comprehensive parameters"""

    @staticmethod
    def calculate_privacy_score(willingness_measure: float, resolution_power: float, beta_coefficient: float) -> float:
        """Calculate privacy score for an attribute"""
        return 1 / exp(beta_coefficient * (1 - willingness_measure) * resolution_power)

    @staticmethod
    def calculate_overall_risk_score(pii_attributes: Dict[str, Set[str]],
                                     weights: Dict[str, float],
                                     willingness_measures: Dict[str, float],
                                     resolution_powers: Dict[str, float],
                                     beta_coefficients: Dict[str, float]) -> float:
        """Calculate overall risk score"""
        overall_risk_score = 0

        if not any(pii_attributes.values()):
            return 0

        for attribute in pii_attributes:
            if pii_attributes[attribute]:  # Only if attribute has values
                weight = weights.get(attribute, 0)
                willingness_measure = willingness_measures.get(attribute, 0)
                resolution_power = resolution_powers.get(attribute, 0)
                beta_coefficient = beta_coefficients.get(attribute, 1)

                privacy_score = RiskCalculator.calculate_privacy_score(
                    willingness_measure, resolution_power, beta_coefficient
                )

                overall_risk_score += weight * privacy_score

        return overall_risk_score

    @staticmethod
    def get_risk_level(risk_score: float) -> str:
        """Get risk level from score"""
        if risk_score == 0:
            return 'No Risk'
        elif risk_score <= 2.74:
            return 'Very Low'
        elif 2.74 < risk_score <= 5.48:
            return 'Low'
        elif 5.48 < risk_score <= 6.87:
            return 'Medium'
        elif 6.87 < risk_score <= 12.25:
            return 'High'
        else:
            return 'Very High'

    @staticmethod
    def get_detailed_risk_analysis(pii_attributes: Dict[str, Set[str]]) -> Dict[str, str]:
        """Provide detailed risk analysis for each category"""
        analysis = {}

        # Categorize PII by risk level
        high_risk_pii = ['SSN', 'Credit Card', 'Passport', 'DDL']
        medium_risk_pii = ['Email', 'Phone', 'Personal Cell', 'Address', 'DOB']
        low_risk_pii = ['Name', 'Location', 'Employer', 'Education', 'Gender']
        social_media_pii = ['Facebook Account', 'Twitter Account', 'Instagram Account', 'LinkedIn Account',
                            'TikTok Account', 'YouTube Account']

        for category, pii_list in [
            ('High Risk', high_risk_pii),
            ('Medium Risk', medium_risk_pii),
            ('Low Risk', low_risk_pii),
            ('Social Media', social_media_pii)
        ]:
            found_items = []
            for pii_type in pii_list:
                if pii_type in pii_attributes and pii_attributes[pii_type]:
                    found_items.append(f"{pii_type} ({len(pii_attributes[pii_type])} found)")

            if found_items:
                analysis[category] = ', '.join(found_items)
            else:
                analysis[category] = 'None found'

        return analysis


# OPTIMIZED: Enhanced URL and content processing functions
async def process_urls_concurrently(urls: List[str], target_name: str, content_processor: ContentProcessor,
                                    apify_manager: APIfyScraperManager, fallback_scraper: OptimizedWebScraper) -> Tuple[
    List[str], List[Dict], Dict]:
    """OPTIMIZED: Process URLs concurrently with controlled parallelism"""

    semaphore = asyncio.Semaphore(CONCURRENT_FETCH_LIMIT)
    all_cleaned_data = []
    extraction_details = []
    scraping_stats = {
        "apify_used": 0,
        "fallback_used": 0,
        "total_failed": 0
    }

    async def process_single_url(url: str, index: int) -> Tuple[Optional[str], Dict]:
        """Process a single URL with semaphore control"""
        async with semaphore:
            try:
                logger.info(f"Processing URL {index + 1}/{len(urls)}: {url}")

                # Fetch content using APIFY with fallback
                html_content = await apify_manager.scraper.fetch_url_with_fallback(url, fallback_scraper)

                if html_content:
                    # Determine scraper used (simple heuristic)
                    if apify_manager.scraper.is_available() and len(html_content) > 1000:
                        scraper_used = "apify"
                    else:
                        scraper_used = "fallback"

                    # Process content with ContentProcessor
                    result = await content_processor.process_webpage(html_content, url, target_name)

                    if result.success and result.content:
                        return html_content, {
                            "source": url,
                            "type": "webpage",
                            "status": "success",
                            "data_points": len(result.content),
                            "scraper_used": scraper_used,
                            "content": result.content
                        }
                    else:
                        return None, {
                            "source": url,
                            "type": "webpage",
                            "status": "no_relevant_content",
                            "data_points": 0,
                            "scraper_used": scraper_used
                        }
                else:
                    return None, {
                        "source": url,
                        "type": "webpage",
                        "status": "fetch_failed",
                        "data_points": 0,
                        "scraper_used": "failed"
                    }

            except Exception as e:
                logger.error(f"Error processing URL {url}: {e}")
                return None, {
                    "source": url,
                    "type": "webpage",
                    "status": "error",
                    "error": str(e),
                    "data_points": 0,
                    "scraper_used": "error"
                }

    # Process all URLs concurrently
    logger.info(f"Starting concurrent processing of {len(urls)} URLs")
    results = await asyncio.gather(
        *[process_single_url(url, i) for i, url in enumerate(urls)],
        return_exceptions=True
    )

    # Process results
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"URL processing failed for {urls[i]}: {result}")
            extraction_details.append({
                "source": urls[i],
                "type": "webpage",
                "status": "exception",
                "error": str(result),
                "data_points": 0,
                "scraper_used": "error"
            })
            scraping_stats["total_failed"] += 1
            continue

        html_content, detail = result
        extraction_details.append(detail)

        # Update stats
        if detail["status"] == "success":
            if detail["scraper_used"] == "apify":
                scraping_stats["apify_used"] += 1
            elif detail["scraper_used"] == "fallback":
                scraping_stats["fallback_used"] += 1

            # Add content to cleaned data
            all_cleaned_data.append(f"Information from webpage: {detail['source']}")
            all_cleaned_data.extend(detail["content"])
            all_cleaned_data.append("---")
        else:
            scraping_stats["total_failed"] += 1

    return all_cleaned_data, extraction_details, scraping_stats


async def process_social_media_concurrently(social_profiles: List[Dict], target_name: str,
                                            content_processor: ContentProcessor, apify_manager: APIfyScraperManager,
                                            fallback_scraper: OptimizedWebScraper) -> Tuple[
    List[str], List[Dict], Dict]:
    """OPTIMIZED: Process social media profiles concurrently"""

    semaphore = asyncio.Semaphore(CONCURRENT_FETCH_LIMIT)
    all_cleaned_data = []
    extraction_details = []
    scraping_stats = {
        "apify_used": 0,
        "fallback_used": 0,
        "total_failed": 0
    }

    async def process_single_social(social_item: Dict, index: int) -> Tuple[Optional[str], Dict]:
        """Process a single social media profile with semaphore control"""
        async with semaphore:
            try:
                url = social_item.get('url', '')
                platform = social_item.get('platform', detect_platform_from_url(url))
                username = social_item.get('username', extract_username_from_url(url))

                if not url:
                    return None, {
                        "source": "",
                        "type": f"social_media_{platform}",
                        "status": "invalid_url",
                        "data_points": 0,
                        "scraper_used": "none"
                    }

                logger.info(f"Processing {platform} profile {index + 1}/{len(social_profiles)}: {url}")

                # Fetch content using APIFY with fallback
                html_content = await apify_manager.scraper.fetch_url_with_fallback(url, fallback_scraper)

                if html_content:
                    scraper_used = "apify" if apify_manager.scraper.is_available() else "fallback"

                    # Process social media content
                    result = await content_processor.process_social_media(
                        html_content, url, platform, username, target_name
                    )

                    if result.success and result.content:
                        return html_content, {
                            "source": url,
                            "type": f"social_media_{platform}",
                            "status": "success",
                            "data_points": len(result.content),
                            "username": username,
                            "scraper_used": scraper_used,
                            "content": result.content
                        }
                    else:
                        return None, {
                            "source": url,
                            "type": f"social_media_{platform}",
                            "status": "no_relevant_content",
                            "data_points": 0,
                            "username": username,
                            "scraper_used": scraper_used
                        }
                else:
                    return None, {
                        "source": url,
                        "type": f"social_media_{platform}",
                        "status": "fetch_failed",
                        "data_points": 0,
                        "username": username,
                        "scraper_used": "failed"
                    }

            except Exception as e:
                logger.error(f"Error processing social media {social_item.get('url', '')}: {e}")
                return None, {
                    "source": social_item.get('url', ''),
                    "type": f"social_media_{social_item.get('platform', 'unknown')}",
                    "status": "error",
                    "error": str(e),
                    "data_points": 0,
                    "scraper_used": "error"
                }

    # Process all social media profiles concurrently
    logger.info(f"Starting concurrent processing of {len(social_profiles)} social media profiles")
    results = await asyncio.gather(
        *[process_single_social(profile, i) for i, profile in enumerate(social_profiles)],
        return_exceptions=True
    )

    # Process results
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"Social media processing failed for {social_profiles[i].get('url', '')}: {result}")
            extraction_details.append({
                "source": social_profiles[i].get('url', ''),
                "type": f"social_media_{social_profiles[i].get('platform', 'unknown')}",
                "status": "exception",
                "error": str(result),
                "data_points": 0,
                "scraper_used": "error"
            })
            scraping_stats["total_failed"] += 1
            continue

        html_content, detail = result
        extraction_details.append(detail)

        # Update stats
        if detail["status"] == "success":
            if detail["scraper_used"] == "apify":
                scraping_stats["apify_used"] += 1
            elif detail["scraper_used"] == "fallback":
                scraping_stats["fallback_used"] += 1

            # Add content to cleaned data
            platform = detail["type"].replace("social_media_", "").capitalize()
            all_cleaned_data.append(f"Social media information from {platform}: {detail['source']}")
            all_cleaned_data.extend(detail["content"])
            all_cleaned_data.append("---")
        else:
            scraping_stats["total_failed"] += 1

    return all_cleaned_data, extraction_details, scraping_stats


# Flask Routes (keeping the same structure)

@app.route('/search', methods=["GET"])
async def search_person():
    """Enhanced search endpoint with Google Custom Search API and comprehensive error handling"""
    # Get parameters with validation
    query = request.args.get('searchName', '').strip()
    include_social = request.args.get('includeSocial', 'true').lower() == 'true'
    max_results = min(int(request.args.get('maxResults', '20')), 50)

    # Input validation
    if not query:
        return jsonify({
            "error": "No search query provided",
            "message": "Please provide a name to search for using the 'searchName' parameter"
        }), 400

    if len(query) < 2:
        return jsonify({
            "error": "Search query too short",
            "message": "Search query must be at least 2 characters long"
        }), 400

    # Log the search request
    logger.info(f"Search request - Query: '{query}', Max results: {max_results}, Include social: {include_social}")

    try:
        # Initialize the enhanced search engine
        async with AdvancedSearchEngine() as search_engine:
            logger.info(f"Starting search for: {query}")

            # Perform web search with Google Custom Search API as primary
            web_results = await search_engine.search_multiple_engines(query, max_results=max_results)

            # Check if we got any results
            if not web_results:
                logger.warning(f"No web results found for query: {query}")
                return jsonify({
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
                }), 404

            # Format the main response
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
            if include_social:
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

            # Add search metadata with Google Custom Search API info
            result["search_metadata"] = {
                "engines_used": ["google_custom_search", "google_direct", "serpapi", "google_library", "bing"],
                "primary_engine": "google_custom_search" if search_engine.google_custom_search.is_available() else "google_direct",
                "google_custom_search_available": search_engine.google_custom_search.is_available(),
                "search_time": datetime.now().isoformat(),
                "include_social_media": include_social
            }

            # Log successful completion
            total_results = len(result["webpages"]) + result.get("total_social_results", 0)
            logger.info(f"Search completed successfully. Total results: {total_results}")

            return jsonify(result)

    except asyncio.TimeoutError:
        logger.error(f"Search timeout for query: {query}")
        return jsonify({
            "error": "Search timeout",
            "message": "The search request took too long to complete. Please try again.",
            "query": query,
            "suggestions": [
                "Try a more specific search term",
                "Check your internet connection",
                "Try again in a few moments"
            ]
        }), 408

    except aiohttp.ClientError as client_error:
        logger.error(f"Network error during search: {client_error}")
        return jsonify({
            "error": "Network error",
            "message": "Unable to connect to search services. Please check your internet connection and try again.",
            "query": query
        }), 503

    except Exception as e:
        # Log the full error for debugging
        import traceback
        error_details = traceback.format_exc()
        logger.error(f"Unexpected error during search for '{query}': {e}")
        logger.error(f"Error details: {error_details}")

        # Return a user-friendly error message
        return jsonify({
            "error": "Internal server error",
            "message": "An unexpected error occurred while searching. Please try again.",
            "query": query,
            "error_type": type(e).__name__,
            "suggestions": [
                "Try again in a few moments",
                "Check if the search term is valid",
                "Contact support if the problem persists"
            ]
        }), 500


@app.route('/extract', methods=["POST"])
async def extract_pii():
    """OPTIMIZED: Enhanced extraction endpoint with concurrent processing"""
    execution_start = time.time()
    logger.info("Starting OPTIMIZED PII extraction process")

    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400

    target_name = data.get('searchName', '')
    selected_urls = data.get('selectedUrls', [])
    selected_social = data.get('selectedSocial', [])

    logger.info(f"Target name: {target_name}")
    logger.info(f"Selected URLs: {len(selected_urls)}")
    logger.info(f"Selected Social Media: {len(selected_social)}")

    if not target_name:
        return jsonify({"error": "No search name provided"}), 400

    if not selected_urls and not selected_social:
        return jsonify({"error": "No URLs or social media profiles selected"}), 400

    try:
        # Initialize processors and scrapers
        content_processor = ContentProcessor(client)
        all_cleaned_data = []
        successful_extractions = 0
        failed_extractions = 0
        extraction_details = []

        # Initialize APIFY scraper manager
        apify_manager = APIfyScraperManager()

        # Track scraping method used
        scraping_stats = {
            "apify_used": 0,
            "fallback_used": 0,
            "total_failed": 0
        }

        # OPTIMIZED: Process URLs and social media concurrently
        async with OptimizedWebScraper() as fallback_scraper:

            # OPTIMIZED: Concurrent URL processing
            if selected_urls:
                logger.info(f"Processing {len(selected_urls)} URLs concurrently...")
                url_data, url_details, url_stats = await process_urls_concurrently(
                    selected_urls, target_name, content_processor, apify_manager, fallback_scraper
                )

                all_cleaned_data.extend(url_data)
                extraction_details.extend(url_details)

                # Update stats
                scraping_stats["apify_used"] += url_stats["apify_used"]
                scraping_stats["fallback_used"] += url_stats["fallback_used"]
                scraping_stats["total_failed"] += url_stats["total_failed"]

                # Count successes and failures
                successful_extractions += sum(1 for detail in url_details if detail["status"] == "success")
                failed_extractions += sum(1 for detail in url_details if detail["status"] != "success")

            # OPTIMIZED: Concurrent social media processing
            if selected_social:
                logger.info(f"Processing {len(selected_social)} social media profiles concurrently...")

                # Convert selected_social format if needed
                social_profiles = []
                for item in selected_social:
                    if isinstance(item, str):
                        social_profiles.append({
                            'url': item,
                            'platform': detect_platform_from_url(item),
                            'username': extract_username_from_url(item)
                        })
                    else:
                        social_profiles.append(item)

                social_data, social_details, social_stats = await process_social_media_concurrently(
                    social_profiles, target_name, content_processor, apify_manager, fallback_scraper
                )

                all_cleaned_data.extend(social_data)
                extraction_details.extend(social_details)

                # Update stats
                scraping_stats["apify_used"] += social_stats["apify_used"]
                scraping_stats["fallback_used"] += social_stats["fallback_used"]
                scraping_stats["total_failed"] += social_stats["total_failed"]

                # Count successes and failures
                successful_extractions += sum(1 for detail in social_details if detail["status"] == "success")
                failed_extractions += sum(1 for detail in social_details if detail["status"] != "success")

        # Remove last separator
        if all_cleaned_data and all_cleaned_data[-1] == "---":
            all_cleaned_data.pop()

        # Get scraping statistics
        apify_stats = apify_manager.get_stats()

        logger.info(f"OPTIMIZED Extraction summary: {successful_extractions} successful, {failed_extractions} failed")
        logger.info(
            f"Scraping stats: APIFY={scraping_stats['apify_used']}, Fallback={scraping_stats['fallback_used']}, Failed={scraping_stats['total_failed']}")

        if not all_cleaned_data:
            return jsonify({
                "message": "No relevant data found about this person from the selected sources.",
                "suggestions": [
                    "Try selecting different URLs that might contain more relevant information",
                    "Check if the selected social media profiles belong to the correct person",
                    "Consider searching with alternative name spellings or variations"
                ],
                "extraction_summary": {
                    'total_sources': len(selected_urls) + len(selected_social),
                    'successful_extractions': successful_extractions,
                    'failed_extractions': failed_extractions,
                    'extraction_time': round(time.time() - execution_start, 2),
                    'scraping_stats': scraping_stats,
                    'apify_performance': apify_stats
                },
                "extraction_details": extraction_details
            })

        logger.info(f"Total cleaned data items: {len(all_cleaned_data)}")

        # Initialize comprehensive PII attributes (SAME AS BEFORE - NO CHANGES)
        attributes = {
            'Name': set(), 'Location': set(), 'Email': set(), 'Phone': set(),
            'DOB': set(), 'Address': set(), 'Gender': set(), 'Employer': set(),
            'Education': set(), 'Birth Place': set(), 'Personal Cell': set(),
            'Business Phone': set(), 'Facebook Account': set(), 'Twitter Account': set(),
            'Instagram Account': set(), 'LinkedIn Account': set(), 'TikTok Account': set(),
            'YouTube Account': set(), 'DDL': set(), 'Passport': set(),
            'Credit Card': set(), 'SSN': set(), 'Family Members': set(),
            'Occupation': set(), 'Salary': set(), 'Website': set()
        }

        # OPTIMIZED: Extract PII using concurrent GPT processing
        pii_extractor = PIIExtractor(client)
        attributes = await pii_extractor.extract_pii(all_cleaned_data, target_name)

        # Convert sets to lists for JSON serialization
        dictionary = {key: list(value) if isinstance(value, set) else value for key, value in attributes.items()}

        # Enhanced risk calculation parameters (SAME AS BEFORE - NO CHANGES)
        weights = {
            'Name': 1, 'Address': 2, 'Location': 1, 'Gender': 1, 'Employer': 2,
            'DOB': 3, 'Education': 1, 'Birth Place': 2, 'Personal Cell': 3,
            'Email': 2, 'Business Phone': 1, 'Facebook Account': 1,
            'Twitter Account': 1, 'Instagram Account': 0.1, 'LinkedIn Account': 1,
            'TikTok Account': 0.1, 'YouTube Account': 0.1, 'DDL': 5,
            'Passport': 5, 'Credit Card': 8, 'SSN': 10, 'Family Members': 2,
            'Occupation': 1, 'Salary': 3, 'Website': 1, 'Phone': 3
        }

        willingness_measures = {
            'Name': 1.0, 'Address': 0.1, 'Location': 0.3, 'Birth Place': 0.2,
            'DOB': 0.4, 'Personal Cell': 0.16, 'Gender': 0.98, 'Employer': 0.7,
            'Education': 0.8, 'Email': 0.5, 'Business Phone': 0.6,
            'Facebook Account': 1.0, 'Twitter Account': 1.0, 'Instagram Account': 1.0,
            'LinkedIn Account': 1.0, 'TikTok Account': 1.0, 'YouTube Account': 1.0,
            'DDL': 0.1, 'Passport': 0.05, 'Credit Card': 0.02, 'SSN': 0.01,
            'Family Members': 0.6, 'Occupation': 0.8, 'Salary': 0.2, 'Website': 0.7,
            'Phone': 0.3
        }

        resolution_powers = {
            'Name': 0.2, 'Address': 0.9, 'Location': 0.3, 'DOB': 0.8,
            'Personal Cell': 0.95, 'Email': 0.9, 'Business Phone': 0.7,
            'Facebook Account': 0.5, 'Twitter Account': 0.5, 'Instagram Account': 0.5,
            'LinkedIn Account': 0.5, 'TikTok Account': 0.5, 'YouTube Account': 0.6,
            'DDL': 1.0, 'Passport': 1.0, 'Credit Card': 1.0, 'SSN': 1.0,
            'Gender': 0.1, 'Employer': 0.5, 'Education': 0.4, 'Birth Place': 0.6,
            'Family Members': 0.4, 'Occupation': 0.3, 'Salary': 0.7, 'Website': 0.5,
            'Phone': 0.9
        }

        beta_coefficients = {key: 1 for key in weights.keys()}

        # Calculate risk score (SAME AS BEFORE - NO CHANGES)
        risk_score = RiskCalculator.calculate_overall_risk_score(
            attributes, weights, willingness_measures, resolution_powers, beta_coefficients
        )

        risk_level = RiskCalculator.get_risk_level(risk_score)
        detailed_analysis = RiskCalculator.get_detailed_risk_analysis(attributes)

        # Add comprehensive results to dictionary
        dictionary['risk_score'] = round(risk_score, 2)
        dictionary['risk_level'] = risk_level
        dictionary['risk_analysis'] = detailed_analysis

        # Calculate PII statistics
        total_pii_found = sum(len(value) for value in attributes.values() if value)
        pii_categories_found = sum(1 for value in attributes.values() if value)

        # Add comprehensive extraction summary with OPTIMIZED stats
        extraction_summary = {
            'total_sources': len(selected_urls) + len(selected_social),
            'webpage_sources': len(selected_urls),
            'social_media_sources': len(selected_social),
            'successful_extractions': successful_extractions,
            'failed_extractions': failed_extractions,
            'total_pii_found': total_pii_found,
            'pii_categories_found': pii_categories_found,
            'data_points_extracted': len(all_cleaned_data),
            'extraction_time': round(time.time() - execution_start, 2),
            'extraction_details': extraction_details,
            'scraping_performance': {
                'apify_used': scraping_stats['apify_used'],
                'fallback_used': scraping_stats['fallback_used'],
                'total_failed': scraping_stats['total_failed'],
                'apify_available': apify_manager.scraper.is_available(),
                'apify_stats': apify_stats,
                'optimization_applied': True,
                'concurrent_processing': True,
                'concurrent_fetch_limit': CONCURRENT_FETCH_LIMIT,
                'concurrent_gpt_limit': CONCURRENT_GPT_LIMIT
            }
        }
        dictionary['extraction_summary'] = extraction_summary

        # Add recommendations based on risk level (SAME AS BEFORE)
        recommendations = []
        if risk_level == 'Very High':
            recommendations = [
                "Immediate action required: Contact platforms to remove sensitive information",
                "Consider identity monitoring services",
                "Review and tighten privacy settings on all online accounts",
                "Be cautious of phishing attempts and identity theft"
            ]
        elif risk_level == 'High':
            recommendations = [
                "Review privacy settings on social media platforms",
                "Consider removing or hiding sensitive information",
                "Monitor your online presence regularly",
                "Be cautious about sharing personal information online"
            ]
        elif risk_level == 'Medium':
            recommendations = [
                "Review what information is publicly available about you",
                "Consider adjusting privacy settings",
                "Monitor your digital footprint periodically",
                "Be cautious about sharing personal information online"
            ]
        elif risk_level == 'Low' or risk_level == 'Very Low':
            recommendations = [
                "Continue practicing good privacy habits",
                "Periodically review your online presence",
                "Be mindful of what you share publicly"
            ]

        dictionary['recommendations'] = recommendations

        execution_time = time.time() - execution_start
        logger.info(f"OPTIMIZED extraction completed in {execution_time:.2f} seconds")
        logger.info(f"Overall Risk Score: {risk_score:.2f} ({risk_level})")
        logger.info(f"Total PII found: {total_pii_found} items across {pii_categories_found} categories")
        logger.info(f"Performance improvement: Concurrent processing enabled")

        return jsonify(dictionary)

    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(f"Extraction error: {e}")
        return jsonify({"error": f"Extraction failed: {str(e)}"}), 500


# Add new health check endpoint for APIFY
@app.route('/apify/health', methods=["GET"])
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
                "concurrent_fetch_limit": CONCURRENT_FETCH_LIMIT,
                "concurrent_gpt_limit": CONCURRENT_GPT_LIMIT,
                "gpt_rate_limit_delay": GPT_RATE_LIMIT_DELAY,
                "url_fetch_timeout": URL_FETCH_TIMEOUT
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
            health_status["message"] = "APIFY test failed, fallback available"

        return jsonify(health_status)

    except Exception as e:
        logger.error(f"APIFY health check failed: {e}")
        return jsonify({
            "status": "degraded",
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "message": "APIFY health check failed, fallback scraper available"
        }), 200  # Return 200 since fallback is available


# Add new endpoint to test APIFY configuration
@app.route('/apify/test', methods=["GET"])
async def test_apify_endpoint():
    """Test APIFY configuration and return detailed results"""
    try:
        test_result = await test_apify_setup()
        test_result["optimization_info"] = {
            "concurrent_processing_enabled": True,
            "concurrent_fetch_limit": CONCURRENT_FETCH_LIMIT,
            "concurrent_gpt_limit": CONCURRENT_GPT_LIMIT,
            "performance_optimizations": [
                "Parallel URL fetching",
                "Concurrent GPT processing",
                "No retry logic (fail fast)",
                "Optimized connection pooling",
                "Rate limiting for API calls"
            ]
        }
        return jsonify(test_result)
    except Exception as e:
        logger.error(f"APIFY test endpoint failed: {e}")
        return jsonify({
            "error": str(e),
            "apify_available": False,
            "test_url_success": False
        }), 500


# OPTIMIZED: Performance monitoring endpoint
@app.route('/performance/stats', methods=["GET"])
async def performance_stats():
    """Get current performance configuration and statistics"""
    return jsonify({
        "optimization_status": "enabled",
        "concurrent_limits": {
            "url_fetching": CONCURRENT_FETCH_LIMIT,
            "gpt_processing": CONCURRENT_GPT_LIMIT
        },
        "timeouts": {
            "url_fetch_timeout": URL_FETCH_TIMEOUT,
            "gpt_rate_limit_delay": GPT_RATE_LIMIT_DELAY
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
    })


# Update the main startup logging
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"Starting OPTIMIZED PII Risk Assessment API v4.0.0 on port {port}")
    logger.info(f"🚀 PERFORMANCE OPTIMIZATIONS ENABLED:")
    logger.info(f"   ⚡ Concurrent URL fetching: {CONCURRENT_FETCH_LIMIT} simultaneous")
    logger.info(f"   ⚡ Concurrent GPT processing: {CONCURRENT_GPT_LIMIT} simultaneous")
    logger.info(f"   ⚡ No retry logic: Fail fast approach")
    logger.info(f"   ⚡ Optimized connection pooling")
    logger.info(f"   ⚡ Rate limiting: {GPT_RATE_LIMIT_DELAY}s between GPT calls")
    logger.info(f"Enhanced scraping: {ENHANCED_SCRAPING}")
    logger.info(f"APIFY Cheerio Scraper: {'Available' if APIfyCheerioScraper().is_available() else 'Not configured'}")
    logger.info(
        f"Google Custom Search API: {'Available' if GoogleCustomSearchAPI().is_available() else 'Not configured'}")
    logger.info(f"SerpAPI available: {SERPAPI_AVAILABLE and bool(os.getenv('SERPAPI_KEY'))}")
    logger.info(f"Google search available: {GOOGLE_SEARCH_AVAILABLE}")
    logger.info(f"Enhanced name matching: Enabled")
    logger.info(f"Multi-engine search: Enabled")
    logger.info(f"Comprehensive PII extraction: Enabled")
    logger.info(f"APIFY fallback mechanism: Enabled")


    # Test APIFY configuration on startup
    async def test_apify_on_startup():
        try:
            test_result = await test_apify_setup()
            if test_result.get("apify_available", False):
                logger.info("✅ APIFY configuration test: PASSED")
            else:
                logger.warning("⚠️  APIFY configuration test: FAILED - Will use fallback scraper")
        except Exception as e:
            logger.error(f"❌ APIFY startup test failed: {e}")


    # Run startup test
    import asyncio

    asyncio.run(test_apify_on_startup())

    # Run the app
    from hypercorn.config import Config
    from hypercorn.asyncio import serve

    config = Config()
    config.bind = [f"0.0.0.0:{port}"]
    config.use_reloader = False

    try:
        asyncio.run(serve(app, config))
    except ImportError:
        # Fallback to Flask's built-in server if hypercorn is not available
        logger.warning("Hypercorn not available, using Flask's built-in server (not recommended for production)")
        app.run(host='0.0.0.0', port=port, debug=False)
