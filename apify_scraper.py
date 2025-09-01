import os
import asyncio
import logging
from typing import Optional, Dict, List
from dataclasses import dataclass
from datetime import datetime
import aiohttp
import json
from urllib.parse import urlparse
import time

# Try to import APIFY client
try:
    from apify_client import ApifyClient

    APIFY_AVAILABLE = True
except ImportError:
    APIFY_AVAILABLE = False
    print("Warning: apify-client not installed. Install with: pip install apify-client")

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class APIfyScrapingResult:
    """Data class for APIFY scraping results"""
    url: str
    html: Optional[str] = None
    title: Optional[str] = None
    text: Optional[str] = None
    success: bool = False
    error: Optional[str] = None
    response_time: float = 0.0
    retry_count: int = 0


class APIfyCheerioScraper:
    """
    APIFY Cheerio Scraper implementation for efficient web scraping

    This class provides individual URL fetching using APIFY's Cheerio Scraper
    with built-in fallback mechanisms and comprehensive error handling.
    """

    def __init__(self):
        """Initialize APIFY Cheerio Scraper"""
        self.api_key = os.getenv("APIFY_API_KEY")
        # Updated to use the correct actor ID from official documentation
        self.actor_id = os.getenv("APIFY_CHEERIO_ACTOR_ID", "YrQuEkowkNCLdk4j2")
        self.timeout = int(os.getenv("APIFY_TIMEOUT", "300"))
        self.max_retries = int(os.getenv("APIFY_MAX_RETRIES", "1"))
        self.enable_proxy = os.getenv("APIFY_ENABLE_PROXY", "true").lower() == "true"
        self.memory_mbytes = int(os.getenv("APIFY_MEMORY_MBYTES", "1024"))

        # Initialize client if available
        self.client = None
        if APIFY_AVAILABLE and self.api_key:
            self.client = ApifyClient(self.api_key)
            logger.info("APIFY Cheerio Scraper initialized successfully")
        else:
            logger.warning("APIFY not available - missing API key or client library")

    def is_available(self) -> bool:
        """Check if APIFY scraper is properly configured and available"""
        return APIFY_AVAILABLE and bool(self.api_key) and self.client is not None

    async def fetch_url(self, url: str) -> Optional[str]:
        """
        Fetch single URL using APIFY Cheerio Scraper

        Args:
            url: URL to scrape

        Returns:
            HTML content as string or None if failed
        """
        if not self.is_available():
            logger.warning("APIFY not available, cannot fetch URL")
            return None

        result = await self.scrape_url_detailed(url)
        return result.html if result.success else None

    async def scrape_url_detailed(self, url: str) -> APIfyScrapingResult:
        """
        Scrape URL with detailed result information

        Args:
            url: URL to scrape

        Returns:
            APIfyScrapingResult with detailed information
        """
        start_time = time.time()

        if not self.is_available():
            return APIfyScrapingResult(
                url=url,
                success=False,
                error="APIFY not available",
                response_time=time.time() - start_time
            )

        # Validate URL
        if not self._is_valid_url(url):
            return APIfyScrapingResult(
                url=url,
                success=False,
                error="Invalid URL format",
                response_time=time.time() - start_time
            )

        # Prepare APIFY input
        run_input = self._prepare_run_input(url)

        retry_count = 0
        last_error = None

        # Retry loop
        while retry_count <= self.max_retries:
            try:
                logger.info(f"APIFY scraping {url} (attempt {retry_count + 1}/{self.max_retries + 1})")

                # Run the actor
                run = await self._run_actor_async(run_input)

                if not run:
                    raise Exception("Failed to start APIFY actor run")

                # Get results - FIXED: Proper handling of APIFY response
                results = await self._get_run_results(run)

                if results:
                    response_time = time.time() - start_time
                    logger.info(f"Successfully scraped {url} in {response_time:.2f}s")

                    return APIfyScrapingResult(
                        url=url,
                        html=results.get('html'),
                        title=results.get('title'),
                        text=results.get('text'),
                        success=True,
                        response_time=response_time,
                        retry_count=retry_count
                    )
                else:
                    raise Exception("No results returned from APIFY")

            except Exception as e:
                last_error = str(e)
                retry_count += 1
                logger.warning(f"APIFY attempt {retry_count} failed for {url}: {e}")

                if retry_count <= self.max_retries:
                    # Exponential backoff
                    wait_time = 2 ** retry_count
                    logger.info(f"Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)

        # All retries failed
        response_time = time.time() - start_time
        logger.error(f"APIFY failed for {url} after {retry_count} attempts: {last_error}")

        return APIfyScrapingResult(
            url=url,
            success=False,
            error=last_error,
            response_time=response_time,
            retry_count=retry_count
        )

    def _prepare_run_input(self, url: str) -> Dict:
        """Prepare APIFY run input configuration - FIXED: Better robots.txt handling"""
        return {
            "startUrls": [{"url": url}],
            "keepUrlFragments": False,
            "respectRobotsTxtFile": False,  # FIXED: Disable robots.txt for better success rate
            "globs": [{"glob": "**"}],
            "pseudoUrls": [],
            "excludes": [{"glob": "/**/*.{png,jpg,jpeg,pdf,gif,svg,ico,css,js}"}],
            "linkSelector": "a[href]",
            "pageFunction": """async function pageFunction(context) {
                const { $, request, log } = context;

                try {
                    // Extract basic page information
                    const pageTitle = $('title').first().text() || '';
                    const url = request.url;

                    // Remove unwanted elements before extracting content
                    $('script, style, nav, header, footer, aside, iframe, noscript').remove();

                    // Extract main content - try different selectors
                    let mainContent = '';
                    const contentSelectors = [
                        'main', 'article', '[role="main"]',
                        '.content', '#content', '.main-content',
                        '.post-content', '.entry-content', '.page-content',
                        '.article-content', '.blog-content', 'body'
                    ];

                    for (const selector of contentSelectors) {
                        const element = $(selector).first();
                        if (element.length > 0) {
                            mainContent = element.text();
                            break;
                        }
                    }

                    // If no specific content area found, use body
                    if (!mainContent) {
                        mainContent = $('body').text();
                    }

                    // Extract metadata
                    const getMetaContent = (name) => {
                        return $(`meta[name="${name}"], meta[property="${name}"]`).attr('content') || '';
                    };

                    const metadata = {
                        description: getMetaContent('description') || getMetaContent('og:description'),
                        keywords: getMetaContent('keywords'),
                        author: getMetaContent('author'),
                        ogTitle: getMetaContent('og:title'),
                        ogDescription: getMetaContent('og:description'),
                        ogSiteName: getMetaContent('og:site_name')
                    };

                    // Get the full HTML content
                    const htmlContent = $.html();

                    log.info('Page scraped successfully', { 
                        url, 
                        titleLength: pageTitle.length,
                        contentLength: mainContent.length,
                        htmlLength: htmlContent.length
                    });

                    return {
                        url: url,
                        title: pageTitle,
                        html: htmlContent,
                        text: mainContent,
                        metadata: metadata,
                        timestamp: new Date().toISOString(),
                        contentLength: htmlContent.length,
                        textLength: mainContent.length
                    };

                } catch (error) {
                    log.error('Error in pageFunction', { url: request.url, error: error.message });

                    // Return minimal data even if there's an error
                    return {
                        url: request.url,
                        title: $('title').first().text() || '',
                        html: $.html() || '',
                        text: $('body').text() || '',
                        metadata: {},
                        timestamp: new Date().toISOString(),
                        error: error.message
                    };
                }
            }""",
            "proxyConfiguration": {
                "useApifyProxy": self.enable_proxy
            } if self.enable_proxy else {},
            "proxyRotation": "RECOMMENDED" if self.enable_proxy else "DISABLED",
            "initialCookies": [],
            "additionalMimeTypes": [],
            "forceResponseEncoding": False,
            "ignoreSslErrors": True,
            "preNavigationHooks": """[]""",
            "postNavigationHooks": """[]""",
            "maxRequestRetries": self.max_retries,
            "maxPagesPerCrawl": 1,
            "maxResultsPerCrawl": 1,
            "maxCrawlingDepth": 0,
            "maxConcurrency": 1,
            "pageLoadTimeoutSecs": 60,
            "pageFunctionTimeoutSecs": 60,
            "debugLog": False,
            "customData": {
                "scraper": "cheerio-enhanced-pii",
                "timestamp": datetime.now().isoformat(),
                "version": "1.0"
            }
        }

    async def _run_actor_async(self, run_input: Dict) -> Optional[Dict]:
        """Run APIFY actor asynchronously"""
        try:
            # Run actor with memory and timeout settings
            run = await asyncio.to_thread(
                self.client.actor(self.actor_id).call,
                run_input=run_input,
                memory_mbytes=self.memory_mbytes,
                timeout_secs=self.timeout
            )
            return run
        except Exception as e:
            logger.error(f"Failed to run APIFY actor: {e}")
            return None

    async def _get_run_results(self, run: Dict) -> Optional[Dict]:
        """Get results from APIFY run - FIXED: Proper handling of ListPage object"""
        try:
            dataset_id = run.get("defaultDatasetId")
            if not dataset_id:
                logger.error("No dataset ID in run results")
                return None

            # Get dataset items - FIXED: Proper handling of APIFY client response
            items_response = await asyncio.to_thread(
                self.client.dataset(dataset_id).list_items
            )

            # FIXED: Handle ListPage object properly
            if hasattr(items_response, 'items'):
                items = items_response.items
            elif isinstance(items_response, dict) and 'items' in items_response:
                items = items_response['items']
            elif isinstance(items_response, list):
                items = items_response
            else:
                logger.error(f"Unexpected items response format: {type(items_response)}")
                return None

            if items and len(items) > 0:
                logger.info(f"Successfully retrieved {len(items)} items from APIFY dataset")
                return items[0]  # Return first (and should be only) result

            logger.warning("No items found in dataset")
            return None

        except Exception as e:
            logger.error(f"Failed to get APIFY results: {e}")
            return None

    def _is_valid_url(self, url: str) -> bool:
        """Validate URL format"""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc]) and result.scheme in ['http', 'https']
        except Exception:
            return False

    async def fetch_url_with_fallback(self, url: str, fallback_scraper) -> Optional[str]:
        """
        Fetch URL with automatic fallback to another scraper

        Args:
            url: URL to scrape
            fallback_scraper: Fallback scraper with fetch_url method

        Returns:
            HTML content or None if both fail
        """
        # Try APIFY first
        if self.is_available():
            html_content = await self.fetch_url(url)
            if html_content:
                logger.info(f"Successfully fetched {url} with APIFY")
                return html_content
            else:
                logger.warning(f"APIFY failed for {url}, trying fallback scraper")
        else:
            logger.info(f"APIFY not available, using fallback scraper for {url}")

        # Use fallback scraper
        try:
            html_content = await fallback_scraper.fetch_url(url)
            if html_content:
                logger.info(f"Successfully fetched {url} with fallback scraper")
                return html_content
            else:
                logger.error(f"Both APIFY and fallback scraper failed for {url}")
                return None
        except Exception as e:
            logger.error(f"Fallback scraper failed for {url}: {e}")
            return None

    async def test_connection(self) -> Dict[str, any]:
        """
        Test APIFY connection and configuration

        Returns:
            Dictionary with test results
        """
        test_result = {
            "apify_available": APIFY_AVAILABLE,
            "api_key_configured": bool(self.api_key),
            "client_initialized": self.client is not None,
            "test_url_success": False,
            "test_error": None,
            "configuration": {
                "actor_id": self.actor_id,
                "timeout": self.timeout,
                "max_retries": self.max_retries,
                "enable_proxy": self.enable_proxy,
                "memory_mbytes": self.memory_mbytes
            }
        }

        if self.is_available():
            # Test with a simple URL
            test_url = "https://httpbin.org/html"
            try:
                result = await self.scrape_url_detailed(test_url)
                test_result["test_url_success"] = result.success
                test_result["test_response_time"] = result.response_time
                if not result.success:
                    test_result["test_error"] = result.error
            except Exception as e:
                test_result["test_error"] = str(e)

        return test_result

    def get_usage_stats(self) -> Dict[str, any]:
        """
        Get APIFY usage statistics (if available)
        Note: This would require additional APIFY API calls to get account usage
        """
        if not self.is_available():
            return {"error": "APIFY not available"}

        # Placeholder for usage stats
        # In a real implementation, you might call APIFY's usage API
        return {
            "note": "Usage stats require additional APIFY API implementation",
            "configured": True,
            "actor_id": self.actor_id,
            "proxy_enabled": self.enable_proxy
        }


class APIfyScraperManager:
    """
    Manager class for handling multiple APIFY scraping operations
    and providing high-level interfaces
    """

    def __init__(self):
        self.scraper = APIfyCheerioScraper()
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "fallback_used": 0,
            "total_response_time": 0.0
        }

    async def scrape_urls_with_progress(self, urls: List[str], fallback_scraper=None,
                                        progress_callback=None) -> Dict[str, Optional[str]]:
        """
        Scrape multiple URLs with progress tracking

        Args:
            urls: List of URLs to scrape
            fallback_scraper: Fallback scraper instance
            progress_callback: Function to call with progress updates

        Returns:
            Dictionary mapping URLs to HTML content (or None if failed)
        """
        results = {}

        for i, url in enumerate(urls, 1):
            if progress_callback:
                progress_callback(i, len(urls), url)

            start_time = time.time()
            self.stats["total_requests"] += 1

            if fallback_scraper:
                html_content = await self.scraper.fetch_url_with_fallback(url, fallback_scraper)
                # FIXED: Better detection of which scraper was used
                if html_content:
                    # Simple heuristic: if APIFY is available and we got content, assume APIFY worked
                    # unless it's a very short response which might indicate fallback
                    if self.scraper.is_available() and len(html_content) > 1000:
                        # Likely APIFY success
                        pass
                    else:
                        self.stats["fallback_used"] += 1
                else:
                    self.stats["fallback_used"] += 1
            else:
                html_content = await self.scraper.fetch_url(url)

            results[url] = html_content

            # Update stats
            response_time = time.time() - start_time
            self.stats["total_response_time"] += response_time

            if html_content:
                self.stats["successful_requests"] += 1
            else:
                self.stats["failed_requests"] += 1

            # Small delay to be respectful
            await asyncio.sleep(0.5)

        return results

    async def _is_apify_content(self, url: str) -> bool:
        """Check if content was fetched by APIFY (simple heuristic)"""
        # This is a simple heuristic - in practice you might want more sophisticated detection
        return self.scraper.is_available()

    def get_stats(self) -> Dict[str, any]:
        """Get scraping statistics"""
        avg_response_time = (self.stats["total_response_time"] / self.stats["total_requests"]
                             if self.stats["total_requests"] > 0 else 0)

        success_rate = (self.stats["successful_requests"] / self.stats["total_requests"] * 100
                        if self.stats["total_requests"] > 0 else 0)

        return {
            **self.stats,
            "average_response_time": round(avg_response_time, 2),
            "success_rate": round(success_rate, 2),
            "apify_available": self.scraper.is_available()
        }

    def reset_stats(self):
        """Reset statistics"""
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "fallback_used": 0,
            "total_response_time": 0.0
        }


# Utility functions for easy integration
async def quick_scrape(url: str, fallback_scraper=None) -> Optional[str]:
    """
    Quick utility function to scrape a single URL

    Args:
        url: URL to scrape
        fallback_scraper: Optional fallback scraper

    Returns:
        HTML content or None
    """
    scraper = APIfyCheerioScraper()

    if fallback_scraper:
        return await scraper.fetch_url_with_fallback(url, fallback_scraper)
    else:
        return await scraper.fetch_url(url)


async def test_apify_setup() -> Dict[str, any]:
    """
    Test APIFY setup and return configuration status

    Returns:
        Dictionary with test results and configuration status
    """
    scraper = APIfyCheerioScraper()
    return await scraper.test_connection()


if __name__ == "__main__":
    # Test the APIFY scraper
    async def test_scraper():
        scraper = APIfyCheerioScraper()

        # Test configuration
        print("Testing APIFY configuration...")
        test_result = await scraper.test_connection()
        print(f"Test result: {test_result}")

        # Test single URL
        if test_result["apify_available"] and test_result["api_key_configured"]:
            print("\nTesting single URL scraping...")
            test_url = "https://example.com"
            html_content = await scraper.fetch_url(test_url)
            print(f"Scraped content length: {len(html_content) if html_content else 0}")
        else:
            print("APIFY not properly configured for testing")


    # Run test
    asyncio.run(test_scraper())
