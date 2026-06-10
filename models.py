from dataclasses import dataclass
from typing import List, Dict, Optional
from pydantic import BaseModel

@dataclass
class SearchResult:
    """Data class for search results"""
    url: str
    title: str
    description: str
    domain: str

@dataclass
class ExtractionResult:
    """Data class for extraction results"""
    source: str
    content: List[str]
    metadata: Dict[str, str]
    success: bool = True
    error: Optional[str] = None

@dataclass
class SearchData:
    """Data class for search results"""
    name : Optional[str]
    age : Optional[str]
    dob : Optional[str]
    gender : Optional[str]
    email : Optional[str]
    public_phone : Optional[str]
    personal_phone: Optional[str]
    address : Optional[str]
    profession : Optional[str]
    organisation : Optional[str]
    government_id : Optional[int]

# Pydantic models for FastAPI request/response validation
class SearchRequest(BaseModel):
    searchName: str
    includeSocial: bool = True
    maxResults: int = 20

class ExtractRequest(BaseModel):
    searchName: str
    selectedUrls: List[str]
    selectedSocial: List[Dict]
    debug: bool = False  # Add debug mode for full PII visibility
    show_pii_preview: bool = True  # Add PII preview toggle

class SearchResponse(BaseModel):
    webpages: List[Dict]
    social_media: Dict
    total_social_results: int
    search_metadata: Dict

class ExtractResponse(BaseModel):
    extracted_data: List[Dict]
    summary: Dict
    processing_time: float
    success: bool

class SocialProfile(BaseModel):
    url: str
    platform: str
    username: str

class PIIAttribute(BaseModel):
    name: str
    values: List[str]
    risk_level: str

class RiskAnalysis(BaseModel):
    overall_risk_score: float
    risk_level: str
    detailed_analysis: Dict[str, str]
    recommendations: List[str]

class ExtractSummary(BaseModel):
    message: str
    suggestions: List[str]
    extraction_summary: Dict
    extraction_details: List[Dict]
