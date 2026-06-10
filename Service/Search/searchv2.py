import os
import logging
import json

from typing import List, Any

import aiohttp
from aiohttp import ClientTimeout, ClientSession

from models import SearchData

logger = logging.getLogger()

class V2:

    def __init__(self):
        self.tavily_api_key = os.getenv("TAVILY_API_KEY")
        self.tavily_base_url = f"{os.getenv("TAVILY_BASE_URL")}/search"
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.openai_base_url = f"{os.getenv("OPENAI_BASE_URL")}/v1/responses"

    async def __aenter__(self):
        connector = aiohttp.TCPConnector(limit=50, limit_per_host=10)
        timeout = ClientTimeout(total=60, connect=10, sock_read=20)
        self.session = ClientSession(connector=connector, timeout=timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()


    def is_available(self) -> bool:
        """Check if envs are properly configured"""
        return bool(self.tavily_api_key and self.openai_api_key)


    async def searchTavily(self, query: str, maxResults: int) -> List:

        logger.info(f"api keys: {self.tavily_api_key}, openai key: {self.openai_api_key}")
        if not self.is_available():
            logger.warning("envs are not configured")
            return []

        request_body = {
            "query" : query,
            "max_results" : maxResults,
            "include_raw_content": True
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.tavily_base_url, json=request_body, headers={"Content-Type": "application/json", "Authorization" : f"Bearer {self.tavily_api_key}"}) as response:
                    if response.status == 200:
                        tavily_results = await response.json()
                        return tavily_results["results"]
                    else :
                        return []
        except Exception as e:
            logger.error(f"Exception fetching tavily search results {e}")
            return []


    async def search_results_to_openai(self, name: str, searchResults: List[Any]) -> List[SearchData]:

        logger.info("Inside search_results_to_openai")

        prompt = f'''
        You are a data extraction engine. 
        
        Task: From the provided content, identify all DISTINCT real-world individuals. with the exact name {name} If the same individual appears multiple times, group all information about that individual into a single record. 
        
        Important constraints: - Extract ONLY information that is explicitly present in the content. - Do NOT infer, guess, or fabricate any values. - Use null for any field that is not present. 
        
        Identity and deduplication rules: - Treat name variants as the SAME individual when they differ only by: - Titles (e.g., Dr., Prof., Mr., Ms.) - Suffixes (e.g., Ph.D., MD) - Punctuation or capitalization - A single individual may have multiple professions, organisations, or addresses over time. - Do NOT create separate individuals solely because: - The organisation differs - The profession differs - Some fields are missing in one mention - Create separate individuals ONLY when the content provides clear conflicting evidence (e.g., clearly different professions in different countries with no shared context). 
        
        Merging rules: - Combine all non-null information for the same individual. - If the same field appears multiple times with different values, keep the most complete value. - Do NOT discard information. 
        
        Output requirements: - Return ONLY valid JSON. - The output must be a JSON array. - Each array item must follow this schema exactly. - Do NOT include explanations, comments, or extra text. Schema: [ {{ "name": "string | "", "age": "string | "", "dob": "string | "", "gender": "string | "", "email": "string | "", "personal phone number": "string | "", "public phone number": "string | "", "address": "string | "", "profession": "string | "", "organisation": "string | "", "ID" : "string | "" }} ] 
        
        Content to analyze:
        '''

        for sr in searchResults:
            content = sr.get("content") or ""
            raw = sr.get("raw_content") or ""
            prompt += f"\n{content}\n{raw}\n"

        request_body = {
            "model" : "gpt-5.1",
            "input" : prompt
        }

        parsed = []

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.openai_base_url, json=request_body, headers={"Authorization" : f"Bearer {self.openai_api_key}", "Content-Type": "application/json"}) as response:
                    if response.status == 200:
                        logger.info(response.json())
                        extractionResults = await response.json()
                        raw_text = extractionResults["output"][0]["content"][0]["text"]
                        if isinstance(raw_text, str):
                            parsed = json.loads(raw_text)
                        else:
                            parsed = raw_text
                    else :
                        return []
        except Exception as e:
            logger.error(f"Exception fetching results from model{e}")
            return []

        logger.info(f"results from open ai{raw_text}")

        search_data_list: List[SearchData] = []

        for item in parsed:
            search_data_list.append(
                SearchData(
                    name=item.get("name"),
                    age=item.get("age"),
                    dob=item.get("dob"),
                    gender=item.get("gender"),
                    email=item.get("email"),
                    public_phone=item.get("public phone number"),
                    personal_phone=item.get("personal phone number"),
                    address=item.get("address"),
                    profession=item.get("profession"),
                    organisation=item.get("organisation"),
                    government_id = item.get("ID")
                )
            )
        return search_data_list











