import os
import asyncio
import aiohttp
from aiohttp import ClientSession, ClientTimeout
import json
import random

from urllib.parse import quote_plus, urlparse
from typing import List, Dict, Set, Tuple, Optional
from bs4 import BeautifulSoup, Comment
import logging
from apify_scraper import APIfyScraperManager
from math import exp

import ast
import re

from models import SearchResult, ExtractionResult
from utils import NameMatcher

logger = logging.getLogger(__name__)

# Configuration
CONCURRENT_FETCH_LIMIT = 10
CONCURRENT_GPT_LIMIT = 5
GPT_RATE_LIMIT_DELAY = 0.2
URL_FETCH_TIMEOUT = 30

# Enhanced scraping flag
try:
    import cloudscraper
    from fake_useragent import UserAgent

    ENHANCED_SCRAPING = True
    ua = UserAgent()
except ImportError:
    ENHANCED_SCRAPING = False
    ua = None

# Fallback user agents
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
]


class GoogleCustomSearchAPI:
    """Google Custom Search API implementation"""

    def __init__(self):
        self.api_key = os.getenv("GOOGLE_CUSTOM_SEARCH_API_KEY")
        self.search_engine_id = os.getenv("GOOGLE_CUSTOM_SEARCH_ENGINE_ID")
        self.base_url = "https://www.googleapis.com/customsearch/v1"

    def is_available(self) -> bool:
        """Check if Google Custom Search API is properly configured"""
        return bool(self.api_key and self.search_engine_id)

    async def search(self, query: str, parameters: List[str], max_results) -> List[SearchResult]:
        """Search using Google Custom Search API"""
        if not self.is_available():
            logger.warning("Google Custom Search API not configured")
            return []

        try:
            results = []
            requests_needed = min((max_results + 9) // 10, 10)

            for page in range(requests_needed):
                start_index = page * 10 + 1
                page_results = await self._search_page(query, parameters, start_index)
                results.extend(page_results)

                if len(results) >= max_results or len(page_results) < 10:
                    break

                if page < requests_needed - 1:
                    await asyncio.sleep(0.1)

            return results[:max_results]

        except Exception as e:
            logger.error(f"Google Custom Search API error: {e}")
            return []

    async def _search_page(self, query: str, parameters: List[str], start_index) -> List[SearchResult]:
        """Search a single page using Google Custom Search API"""

        results: List[SearchResult] = []

        if parameters:
            for param in parameters:
                params = {
                    'key': self.api_key,
                    'cx': self.search_engine_id,
                    'q': query,
                    'start': start_index,
                    'num': 10,
                    'safe': 'medium',
                    'fields': 'items(title,link,snippet,displayLink),searchInformation(totalResults)',
                    'exactTerms': param
                }

                logger.info(f"Params before {params}")

                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(self.base_url, params=params) as response:
                            if response.status == 200:
                                data = await response.json()
                                self._parse_custom_search_results(data, results)
                            elif response.status == 429:
                                logger.warning("Google Custom Search API rate limit exceeded")
                            elif response.status == 403:
                                logger.warning("Google Custom Search API quota exceeded or invalid credentials")
                            else:
                                logger.warning(f"Google Custom Search API returned status {response.status}")
                except asyncio.TimeoutError:
                    logger.error("Google Custom Search API timeout")
                except Exception as e:
                    logger.error(f"Error in Google Custom Search API request: {e}")

        if parameters:
            return results

        logging.info("No Results found for specific parameters now searching with the query")

        results: List[SearchResult] = []

        params = {
            'key': self.api_key,
            'cx': self.search_engine_id,
            'q': query,
            'start': start_index,
            'num': 10,
            'safe': 'medium',
            'fields': 'items(title,link,snippet,displayLink),searchInformation(totalResults)'
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        self._parse_custom_search_results(data, results)
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

        return results

    def _parse_custom_search_results(self, data: dict, results : List[SearchResult]):
        """Parse Google Custom Search API response"""

        logger.info(f"Parsing results for {data}")


        items = data.get('items', [])

        for item in items:
            try:
                url = item.get('link', '')
                title = item.get('title', '')
                snippet = item.get('snippet', '')
                display_link = item.get('displayLink', '')

                if not self._is_valid_url(url):
                    continue

                results.append(SearchResult(
                    url=url,
                    title=title,
                    description=snippet,
                    domain=display_link or urlparse(url).netloc
                ))
            except Exception as e:
                logger.warning(f"Error parsing search result: {e}")
                continue

        logger.info(f"Google Custom Search API  Results parsing {results}")

    def _is_valid_url(self, url: str) -> bool:
        """Validate if URL is worth including"""
        if not url:
            return False

        invalid_schemes = ['javascript:', 'mailto:', 'tel:', 'data:', '#']
        if any(url.lower().startswith(scheme) for scheme in invalid_schemes):
            return False

        if not url.startswith(('http://', 'https://')):
            return False

        unwanted_domains = [
            'google.com/search', 'google.com/url', 'webcache.googleusercontent.com',
            'accounts.google.com', 'support.google.com', 'policies.google.com',
            'translate.google.com', 'maps.google.com'
        ]

        if any(domain in url.lower() for domain in unwanted_domains):
            return False

        return True


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
            limit=50,
            limit_per_host=10,
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
        if ENHANCED_SCRAPING and ua is not None:
            try:
                user_agent = ua.random
            except:
                user_agent = random.choice(USER_AGENTS)
        else:
            user_agent = random.choice(USER_AGENTS)

        return {
            'User-Agent': user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Cache-Control': 'max-age=0',
        }

    async def fetch_url(self, url: str) -> Optional[str]:
        """Fetch single URL with optimized approach"""
        if not self.session:
            return None

        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    content = await response.text()
                    return content
                else:
                    logger.warning(f"HTTP {response.status} for {url}")
                    return None

        except asyncio.TimeoutError:
            logger.warning(f"Timeout fetching {url}")
            return None
        except Exception as e:
            logger.warning(f"Error fetching {url}: {e}")
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


class AdvancedSearchEngine:
    """Enhanced search engine with Google Custom Search API integration"""

    def __init__(self):
        self.session = None
        self.google_custom_search = GoogleCustomSearchAPI()

        # Enhanced user agents for better success rate
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        ]

    async def __aenter__(self):
        connector = aiohttp.TCPConnector(limit=50, limit_per_host=10)
        timeout = ClientTimeout(total=30, connect=10, sock_read=20)
        self.session = ClientSession(connector=connector, timeout=timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    def _get_headers(self):
        """Get headers with random user agent"""
        return {
            'User-Agent': random.choice(self.user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Cache-Control': 'max-age=0',
        }

    async def search_multiple_engines(self, query: str, parameters: str, max_results) -> List[SearchResult]:
        """Search using Google Custom Search API only"""
        all_results = []

        if parameters:
            parameters = parameters.split(",")

        logger.info(f"Searching with {parameters}")

        # Primary: Google Custom Search API (most reliable)
        if self.google_custom_search.is_available():
            try:
                custom_search_results = await self.google_custom_search.search(query, parameters, max_results)
                all_results.extend(custom_search_results)
                logger.info(f"Google Custom Search API returned {len(custom_search_results)} results")
            except Exception as e:
                logger.error(f"Google Custom Search API failed: {e}")

        logger.info(f"Google Custom Search API returned {len(all_results)} results")
        return all_results

    async def _search_google_direct(self, query: str, max_results: int) -> List[SearchResult]:
        """Direct Google search using BeautifulSoup - Fallback method"""
        if not self.session:
            return []

        try:
            headers = self._get_headers()
            search_url = f"https://www.google.com/search?q={quote_plus(query)}&num={max_results}"

            async with self.session.get(search_url, headers=headers) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    return self._parse_google_results(soup, query, max_results)
                else:
                    logger.warning(f"Google search failed with status {response.status}")
                    return []

        except Exception as e:
            logger.error(f"Google search error: {e}")
            return []

    def _parse_google_results(self, soup: BeautifulSoup, query: str, max_results: int = 25) -> List[SearchResult]:
        """Parse Google search results from HTML"""
        try:
            results = []
            result_containers = soup.find_all('div', class_='g')

            for container in result_containers[:max_results]:
                try:
                    title_elem = container.find('h3')
                    if not title_elem:
                        continue

                    link_elem = title_elem.find('a')
                    if not link_elem or not link_elem.get('href'):
                        continue

                    url = link_elem['href']
                    title = link_elem.get_text().strip()

                    desc_elem = container.find('span', class_='aCOpRe')
                    description = desc_elem.get_text().strip() if desc_elem else ""

                    if not self._is_valid_url(url):
                        continue

                    results.append(SearchResult(
                        url=url,
                        title=title,
                        description=description,
                        domain=urlparse(url).netloc
                    ))

                except Exception as e:
                    logger.warning(f"Error parsing Google result: {e}")
                    continue

            return results

        except Exception as e:
            logger.error(f"Error parsing Google results: {e}")
            return []

    def _is_valid_url(self, url: str) -> bool:
        """Validate if URL is worth including"""
        if not url:
            return False

        invalid_schemes = ['javascript:', 'mailto:', 'tel:', 'data:', '#']
        if any(url.lower().startswith(scheme) for scheme in invalid_schemes):
            return False

        if not url.startswith(('http://', 'https://')):
            return False

        unwanted_domains = [
            'google.com/search', 'google.com/url', 'webcache.googleusercontent.com',
            'accounts.google.com', 'support.google.com', 'policies.google.com',
            'translate.google.com', 'maps.google.com'
        ]

        if any(domain in url.lower() for domain in unwanted_domains):
            return False

        return True

    def _calculate_relevance(self, query: str, title: str, description: str, url: str = "") -> int:
        """Calculate relevance score for search results"""
        score = 0
        query_terms = query.lower().split()

        title_lower = title.lower() if title else ""
        description_lower = description.lower() if description else ""
        url_lower = url.lower() if url else ""

        for term in query_terms:
            if term in title_lower:
                score += 15
            if term in description_lower:
                score += 5
            if term in url_lower:
                score += 3

        if query.lower() in title_lower:
            score += 25
        if query.lower() in description_lower:
            score += 10

        if url:
            high_authority_domains = [
                'wikipedia.org', 'linkedin.com', 'facebook.com', 'twitter.com',
                'instagram.com', 'youtube.com', 'github.com', 'stackoverflow.com',
                'medium.com', 'blogspot.com', 'wordpress.com'
            ]

            domain = urlparse(url).netloc.lower()
            if any(auth_domain in domain for auth_domain in high_authority_domains):
                score += 10

        if any(term.istitle() for term in query.split()):
            score += 5

        return max(score, 1)

    def _deduplicate_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """Remove duplicate URLs and normalize domains"""
        seen_urls = set()
        unique_results = []

        for result in results:
            normalized_url = result.url.lower().rstrip('/')
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
        print("inside find_social_accounts")
        platforms = [
            "facebook",
            "twitter",
            "instagram",
            "linkedin",
            "tiktok",
            "youtube"
        ]

        social_results = {}

        for platform in platforms:
            try:
                results = await self.search_engine.search_multiple_engines(name, f"site:{platform}.com", max_results=3)
                logger.info(f"Result for platform :{platform}")
                social_results[platform] = results
            except Exception as e:
                logger.error(f"Error searching {platform} for {name}: {e}")
                social_results[platform] = []
        return social_results


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
        from utils import extract_username_from_url
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
        elif risk_score <= 2.75:
            return 'Very Low'
        elif 2.75 < risk_score <= 5.48:
            return 'Low'
        elif 5.48 < risk_score <= 9:
            return 'Medium'
        elif 10 < risk_score <= 12.25:
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
                from utils import detect_platform_from_url, extract_username_from_url
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


# NEW SIMPLIFIED EXTRACTION FUNCTIONS

async def fetch_all_urls_parallel(all_urls: List[str], apify_manager: APIfyScraperManager,
                                  fallback_scraper: OptimizedWebScraper) -> List[Dict]:
    """Fetch content from all URLs asynchronously"""

    semaphore = asyncio.Semaphore(CONCURRENT_FETCH_LIMIT)  # Max 10 concurrent

    async def fetch_single_url(url: str) -> Dict:
        async with semaphore:
            try:
                logger.info(f"Fetching content from: {url}")

                # Use APIFY + fallback scraping
                html_content = await apify_manager.scraper.fetch_url_with_fallback(url, fallback_scraper)

                if html_content:
                    # Clean content with BeautifulSoup
                    soup = BeautifulSoup(html_content, 'html.parser')

                    # Remove unwanted elements
                    for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'iframe']):
                        element.decompose()

                    # Remove comments
                    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
                        comment.extract()

                    # Extract clean text content
                    clean_content = soup.get_text()

                    # Clean up the text
                    lines = clean_content.split('\n')
                    cleaned_lines = [line.strip() for line in lines if line.strip() and len(line.strip()) > 3]
                    final_content = ' '.join(cleaned_lines)

                    # Detect platform for social media
                    from utils import detect_platform_from_url
                    platform = detect_platform_from_url(url)
                    if platform == 'unknown':
                        platform = 'webpage'

                    return {
                        "url": url,
                        "status": "success",
                        "content": final_content[:15000],  # Limit content size to avoid token issues
                        "content_length": len(final_content),
                        "platform": platform
                    }
                else:
                    # Detect platform even for failed cases
                    from utils import detect_platform_from_url
                    platform = detect_platform_from_url(url)
                    if platform == 'unknown':
                        platform = 'webpage'

                    return {
                        "url": url,
                        "status": "failed",
                        "content": "",
                        "error": "Failed to fetch content",
                        "platform": platform
                    }

            except Exception as e:
                logger.error(f"Error fetching {url}: {e}")
                # Detect platform even for error cases
                from utils import detect_platform_from_url
                platform = detect_platform_from_url(url)
                if platform == 'unknown':
                    platform = 'webpage'

                return {
                    "url": url,
                    "status": "error",
                    "content": "",
                    "error": str(e),
                    "platform": platform
                }

    # Process all URLs in parallel
    logger.info(f"Starting parallel fetch of {len(all_urls)} URLs")
    results = await asyncio.gather(*[fetch_single_url(url) for url in all_urls])

    successful = sum(1 for r in results if r["status"] == "success")
    logger.info(f"Parallel fetch completed: {successful}/{len(all_urls)} successful")

    return results


async def extract_pii_from_each_page(fetch_results: List[Dict], target_name: str, openai_client) -> Dict[str, set]:
    """Extract PII from each page sequentially to avoid token limits"""

    # Initialize final PII attributes
    final_attributes = {
        'Name': set(), 'Location': set(), 'Email': set(), 'Phone': set(),
        'DOB': set(), 'Address': set(), 'Gender': set(), 'Employer': set(),
        'Education': set(), 'Birth Place': set(), 'Personal Cell': set(),
        'Business Phone': set(), 'Facebook Account': set(), 'Twitter Account': set(),
        'Instagram Account': set(), 'LinkedIn Account': set(), 'TikTok Account': set(),
        'YouTube Account': set(), 'DDL': set(), 'Passport': set(),
        'Credit Card': set(), 'SSN': set(), 'Family Members': set(),
        'Occupation': set(), 'Salary': set(), 'Website': set()
    }

    successful_extractions = 0
    failed_extractions = 0

    # Process each successful fetch result
    for result in fetch_results:
        if result["status"] == "success" and result["content"]:
            try:
                logger.info(f"Extracting PII from: {result['url']} ({result['platform']})")

                # Extract PII from single page
                page_pii = await extract_pii_from_single_page(
                    result["content"],
                    result["url"],
                    target_name,
                    result.get("platform", "webpage"),
                    openai_client
                )

                # Append to final attributes
                for key, values in page_pii.items():
                    if key in final_attributes and values:
                        if isinstance(values, (list, set)):
                            final_attributes[key].update(values)
                        elif values and str(values).strip():
                            final_attributes[key].add(str(values))

                successful_extractions += 1
                logger.info(f"Successfully extracted PII from {result['url']}")
                logger.info(f"PII extracted from {result['url']}: {[(k, len(v)) for k, v in page_pii.items() if v]}")

                # Rate limiting between GPT calls
                await asyncio.sleep(GPT_RATE_LIMIT_DELAY)

            except Exception as e:
                logger.error(f"Failed to extract PII from {result['url']}: {e}")
                failed_extractions += 1
        else:
            failed_extractions += 1

    logger.info(f"PII extraction completed: {successful_extractions} successful, {failed_extractions} failed")
    return final_attributes


async def extract_pii_from_single_page(content: str, url: str, target_name: str, platform: str, openai_client) -> Dict[
    str, set]:
    """Extract PII from a single page using GPT"""

    # Initialize empty result
    empty_result = {
        'Name': set(), 'Location': set(), 'Email': set(), 'Phone': set(),
        'DOB': set(), 'Address': set(), 'Gender': set(), 'Employer': set(),
        'Education': set(), 'Birth Place': set(), 'Personal Cell': set(),
        'Business Phone': set(), 'Facebook Account': set(), 'Twitter Account': set(),
        'Instagram Account': set(), 'LinkedIn Account': set(), 'TikTok Account': set(),
        'YouTube Account': set(), 'DDL': set(), 'Passport': set(),
        'Credit Card': set(), 'SSN': set(), 'Family Members': set(),
        'Occupation': set(), 'Salary': set(), 'Website': set()
    }

    # Create page-specific prompt
    if platform != "webpage":
        system_prompt = f"Extract personal information from this {platform} page for '{target_name}'. Be precise and only extract information clearly related to this person."
        context = f"This is a {platform} profile page."
    else:
        system_prompt = f"Extract personal information from this webpage for '{target_name}'. Be precise and only extract information clearly related to this person."
        context = "This is a regular webpage."

    user_prompt = f"""
    {context}
    URL: {url}
    Target Person: {target_name}
    
    Content:
    {content}
    
    Extract all Personally Identifiable Information (PII) about '{target_name}' from the above content.
    Only include information that clearly relates to this specific person.
    
    Return valid JSON with these exact field names (use empty string "" if not found):
    {{
        "Name": "full names found",
        "Location": "locations found", 
        "Email": "email addresses found",
        "Phone": "phone numbers found",
        "DOB": "date of birth info found",
        "Address": "physical addresses found",
        "Gender": "gender information found",
        "Employer": "employer/company found",
        "Education": "education background found",
        "Birth Place": "place of birth found",
        "Personal Cell": "mobile numbers found",
        "Business Phone": "work phone numbers found",
        "Facebook Account": "facebook profiles found",
        "Twitter Account": "twitter profiles found", 
        "Instagram Account": "instagram profiles found",
        "LinkedIn Account": "linkedin profiles found",
        "TikTok Account": "tiktok profiles found",
        "YouTube Account": "youtube profiles found",
        "DDL": "driver license info found",
        "Passport": "passport info found",
        "Credit Card": "credit card info found",
        "SSN": "social security numbers found",
        "Family Members": "family members found",
        "Occupation": "job titles found",
        "Salary": "income information found",
        "Website": "personal websites found"
    }}
    
    For multiple values, separate with commas.
    """

    try:
        response = await asyncio.to_thread(
            openai_client.chat.completions.create,
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=2000  # Smaller limit for single page
        )

        result = response.choices[0].message.content.strip()

        # Clean JSON response
        if result.startswith('```json'):
            result = result[7:]
        if result.startswith('```'):
            result = result[3:]
        if result.endswith('```'):
            result = result[:-3]

        result = result.strip()

        # Parse JSON
        try:
            pii_data = json.loads(result)
        except json.JSONDecodeError:
            try:
                pii_data = ast.literal_eval(result)
            except:
                logger.error(f"Failed to parse GPT response for {url}")
                return empty_result

        # Convert to sets and validate
        validated_pii = {}
        for key, value in pii_data.items():
            if key in empty_result and value and str(value).strip() and str(value) != "":
                value_str = str(value).strip()
                if ',' in value_str:
                    # Multiple values separated by commas
                    values = [v.strip() for v in value_str.split(',') if v.strip()]
                    validated_pii[key] = set(values)
                else:
                    validated_pii[key] = {value_str}
            else:
                validated_pii[key] = set()

        # Log what was found
        found_items = sum(len(v) for v in validated_pii.values() if v)
        if found_items > 0:
            logger.info(f"Found {found_items} PII items from {url}")

        return validated_pii

    except Exception as e:
        logger.error(f"GPT extraction failed for {url}: {e}")
        return empty_result
