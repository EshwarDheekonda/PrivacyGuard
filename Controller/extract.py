from fastapi import APIRouter
import time
import logging
from datetime import datetime
import os
from openai import OpenAI

from core import (
    OptimizedWebScraper, ContentProcessor, PIIExtractor, RiskCalculator,
    process_urls_concurrently, process_social_media_concurrently
)
from apify_scraper import APIfyScraperManager
from models import ExtractRequest, ExtractResponse, ExtractSummary
from utils import detect_platform_from_url, extract_username_from_url

logger = logging.getLogger(__name__)
router = APIRouter()

def mask_sensitive_pii(pii_data: dict) -> dict:
    """Mask sensitive PII for safe logging"""
    masked_data = {}
    sensitive_fields = ['Email', 'Phone', 'Personal Cell', 'Business Phone', 'SSN', 'Credit Card']
    
    for key, values in pii_data.items():
        if key in sensitive_fields and values:
            if isinstance(values, list):
                masked_values = []
                for value in values:
                    if key == 'Email':
                        # Mask email: john@example.com -> j***@e***.com
                        if '@' in value:
                            local, domain = value.split('@', 1)
                            if len(local) > 1 and len(domain) > 1:
                                masked_values.append(f"{local[0]}***@{domain[0]}***.{domain.split('.')[-1]}")
                            else:
                                masked_values.append("***@***.***")
                        else:
                            masked_values.append("***@***.***")
                    elif key in ['Phone', 'Personal Cell', 'Business Phone']:
                        # Mask phone: +1234567890 -> +1***-***-***0
                        digits = ''.join(filter(str.isdigit, value))
                        if len(digits) >= 4:
                            masked_values.append(f"{digits[:1]}***-***-***{digits[-1]}")
                        else:
                            masked_values.append("***-***-****")
                    elif key == 'SSN':
                        # Mask SSN: 123-45-6789 -> ***-**-***9
                        digits = ''.join(filter(str.isdigit, value))
                        if len(digits) >= 4:
                            masked_values.append(f"***-**-***{digits[-1]}")
                        else:
                            masked_values.append("***-**-****")
                    elif key == 'Credit Card':
                        # Mask credit card: 1234-5678-9012-3456 -> ****-****-****-3456
                        digits = ''.join(filter(str.isdigit, value))
                        if len(digits) >= 4:
                            masked_values.append(f"****-****-****-{digits[-4:]}")
                        else:
                            masked_values.append("****-****-****-****")
                    else:
                        masked_values.append("***")
                masked_data[key] = masked_values
            else:
                masked_data[key] = ["***"]
        else:
            masked_data[key] = values
    
    return masked_data

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@router.post("/extract")
async def extract_pii(request: ExtractRequest):
    """SIMPLIFIED: New extraction process with individual page processing"""
    execution_start = time.time()
    logger.info("Starting SIMPLIFIED PII extraction process")

    target_name = request.searchName
    selected_urls = request.selectedUrls
    selected_social = request.selectedSocial
    debug_mode = request.debug
    show_pii_preview = request.show_pii_preview

    logger.info(f"Target name: {target_name}")
    logger.info(f"Selected URLs: {len(selected_urls)}")
    logger.info(f"Selected Social Media: {len(selected_social)}")
    logger.info(f"Debug mode: {debug_mode}")
    logger.info(f"Show PII preview: {show_pii_preview}")

    if not target_name:
        return {"error": "No search name provided"}

    if not selected_urls and not selected_social:
        return {"error": "No URLs or social media profiles selected"}

    try:
        # PHASE 1: Consolidate all URLs
        all_urls = selected_urls.copy()
        for social in selected_social:
            if isinstance(social, str):
                all_urls.append(social)
            else:
                all_urls.append(social.get('url', ''))
        
        logger.info(f"Processing {len(all_urls)} URLs total")
        
        # PHASE 2: Fetch content from all URLs asynchronously
        async with OptimizedWebScraper() as fallback_scraper:
            apify_manager = APIfyScraperManager()
            
            logger.info("Fetching content from all URLs asynchronously...")
            from core import fetch_all_urls_parallel
            fetch_results = await fetch_all_urls_parallel(all_urls, apify_manager, fallback_scraper)
            
            successful_fetches = sum(1 for r in fetch_results if r["status"] == "success")
            failed_fetches = len(fetch_results) - successful_fetches
            
            logger.info(f"Content fetching completed: {successful_fetches} successful, {failed_fetches} failed")
            
            if successful_fetches == 0:
                return {
                    "query": target_name,
                    "timestamp": datetime.now().isoformat(),
                    "message": "Failed to fetch content from any of the provided URLs.",
                    "suggestions": [
                        "Check if the URLs are accessible",
                        "Try with different URLs",
                        "Check your internet connection"
                    ],
                    "extraction_summary": {
                        "total_urls": len(all_urls),
                        "successful_fetches": 0,
                        "failed_fetches": failed_fetches,
                        "extraction_time": round(time.time() - execution_start, 2)
                    }
                }
            
            # PHASE 3: Extract PII from each page sequentially
            logger.info("Extracting PII from each page sequentially...")
            from core import extract_pii_from_each_page
            final_attributes = await extract_pii_from_each_page(fetch_results, target_name, client)
            
            # CRITICAL FIX: Initialize with ALL PII attributes first (matching Flask exactly)
            dictionary = {
                'Name': [], 'Location': [], 'Email': [], 'Phone': [],
                'DOB': [], 'Address': [], 'Gender': [], 'Employer': [],
                'Education': [], 'Birth Place': [], 'Personal Cell': [],
                'Business Phone': [], 'Facebook Account': [], 'Twitter Account': [],
                'Instagram Account': [], 'LinkedIn Account': [], 'TikTok Account': [],
                'YouTube Account': [], 'DDL': [], 'Passport': [],
                'Credit Card': [], 'SSN': [], 'Family Members': [],
                'Occupation': [], 'Salary': [], 'Website': []
            }
            
            # FIXED: Properly convert sets to lists and ensure all attributes are included
            logger.info(f"Converting final_attributes to dictionary format...")
            logger.info(f"Final attributes before conversion: {[(k, len(v) if isinstance(v, (set, list)) else v) for k, v in final_attributes.items() if v]}")
            
            for key, value in final_attributes.items():
                if key in dictionary:
                    if isinstance(value, set):
                        dictionary[key] = list(value) if value else []
                    elif isinstance(value, list):
                        dictionary[key] = value
                    else:
                        dictionary[key] = [value] if value else []
                else:
                    # If key not in dictionary, add it
                    if isinstance(value, set):
                        dictionary[key] = list(value) if value else []
                    elif isinstance(value, list):
                        dictionary[key] = value
                    else:
                        dictionary[key] = [value] if value else []
            
            logger.info(f"Dictionary after conversion: {[(k, len(v) if isinstance(v, list) else v) for k, v in dictionary.items() if isinstance(v, list) and v]}")

        # Calculate risk scores using the converted dictionary
        logger.info("Calculating risk scores...")
        
        # Convert dictionary back to sets for risk calculation
        pii_attributes_for_risk = {}
        for key, value in dictionary.items():
            if key in ['Name', 'Location', 'Email', 'Phone', 'DOB', 'Address', 'Gender', 'Employer',
                      'Education', 'Birth Place', 'Personal Cell', 'Business Phone', 'Facebook Account',
                      'Twitter Account', 'Instagram Account', 'LinkedIn Account', 'TikTok Account',
                      'YouTube Account', 'DDL', 'Passport', 'Credit Card', 'SSN', 'Family Members',
                      'Occupation', 'Salary', 'Website']:
                pii_attributes_for_risk[key] = set(value) if value else set()
        
        # Define risk parameters (EXACT COPY FROM ORIGINAL FLASK)
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

        # Calculate risk score using the converted attributes
        overall_risk_score = RiskCalculator.calculate_overall_risk_score(
            pii_attributes_for_risk, weights, willingness_measures, resolution_powers, beta_coefficients
        )
        risk_level = RiskCalculator.get_risk_level(overall_risk_score)
        detailed_analysis = RiskCalculator.get_detailed_risk_analysis(pii_attributes_for_risk)

        # Add recommendations based on risk level (EXACT COPY FROM ORIGINAL)
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

        # Add comprehensive results to dictionary (EXACT COPY FROM ORIGINAL)
        dictionary['risk_score'] = round(overall_risk_score, 2)
        dictionary['risk_level'] = risk_level
        dictionary['risk_analysis'] = detailed_analysis

        # Calculate PII statistics using the dictionary
        total_pii_found = sum(len(value) for value in dictionary.values() if isinstance(value, list) and value)
        pii_categories_found = sum(1 for value in dictionary.values() if isinstance(value, list) and value)
        
        # Create extraction_details array (matching Flask format)
        extraction_details = []
        for result in fetch_results:
            # Safe access to platform field with default value
            platform = result.get("platform", "webpage")
            detail = {
                "source": result["url"],
                "type": "social_media" if platform != "webpage" else "webpage",
                "status": result["status"],
                "data_points": len(result.get("content", "")),
                "platform": platform,
                "scraper_used": "apify" if platform != "webpage" and result["status"] == "success" else ("fallback" if result["status"] == "success" else "failed")
            }
            extraction_details.append(detail)
        
        # Add comprehensive extraction summary in the EXACT original Flask format
        extraction_summary = {
            'total_sources': len(selected_urls) + len(selected_social),
            'webpage_sources': len(selected_urls),
            'social_media_sources': len(selected_social),
            'successful_extractions': successful_fetches,  # Match Flask naming
            'failed_extractions': failed_fetches,          # Match Flask naming
            'total_pii_found': total_pii_found,
            'pii_categories_found': pii_categories_found,
            'data_points_extracted': len(fetch_results),   # Add this Flask field
            'extraction_time': round(time.time() - execution_start, 2),
            'extraction_details': extraction_details,      # Add detailed results
            'scraping_performance': {                      # Add performance stats (Flask format)
                'apify_used': sum(1 for r in fetch_results if r.get("platform", "webpage") != "webpage" and r["status"] == "success"),
                'fallback_used': sum(1 for r in fetch_results if r.get("platform", "webpage") == "webpage" and r["status"] == "success"),
                'total_failed': failed_fetches,
                'apify_available': apify_manager.scraper.is_available(),
                'apify_stats': {
                    'requests_made': sum(1 for r in fetch_results if r.get("platform", "webpage") != "webpage"),
                    'successful_requests': sum(1 for r in fetch_results if r.get("platform", "webpage") != "webpage" and r["status"] == "success"),
                    'failed_requests': sum(1 for r in fetch_results if r.get("platform", "webpage") != "webpage" and r["status"] != "success")
                },
                'optimization_applied': True,
                'concurrent_processing': True,
                'concurrent_fetch_limit': 10,
                'concurrent_gpt_limit': 5
            }
        }
        dictionary['extraction_summary'] = extraction_summary
        
        # Add query and timestamp fields to match original format
        dictionary['query'] = target_name
        dictionary['timestamp'] = datetime.now().isoformat()
        
        # Add recommendations in the original format
        dictionary['recommendations'] = recommendations
        
        # Add message in the original format
        dictionary['message'] = f"PII extraction completed successfully. Found {total_pii_found} items of personal information across {pii_categories_found} categories."
        
        # Add suggestions in the original format
        dictionary['suggestions'] = recommendations

        # Add PII preview for debugging (masked sensitive data)
        pii_preview = {}
        for key in ['Name', 'Location', 'Email', 'Phone', 'Address', 'Employer', 'Education', 'DOB', 'Gender']:
            if key in dictionary and dictionary[key]:
                if key in ['Email', 'Phone', 'Personal Cell', 'Business Phone', 'SSN', 'Credit Card']:
                    # Mask sensitive data in response
                    masked_values = []
                    for value in dictionary[key]:
                        if key == 'Email' and '@' in value:
                            local, domain = value.split('@', 1)
                            if len(local) > 1 and len(domain) > 1:
                                masked_values.append(f"{local[0]}***@{domain[0]}***.{domain.split('.')[-1]}")
                            else:
                                masked_values.append("***@***.***")
                        elif key in ['Phone', 'Personal Cell', 'Business Phone']:
                            digits = ''.join(filter(str.isdigit, value))
                            if len(digits) >= 4:
                                masked_values.append(f"{digits[:1]}***-***-***{digits[-1]}")
                            else:
                                masked_values.append("***-***-****")
                        elif key == 'SSN':
                            digits = ''.join(filter(str.isdigit, value))
                            if len(digits) >= 4:
                                masked_values.append(f"***-**-***{digits[-1]}")
                            else:
                                masked_values.append("***-**-****")
                        elif key == 'Credit Card':
                            digits = ''.join(filter(str.isdigit, value))
                            if len(digits) >= 4:
                                masked_values.append(f"****-****-****-{digits[-4:]}")
                            else:
                                masked_values.append("****-****-****-****")
                        else:
                            masked_values.append("***")
                    pii_preview[key] = masked_values
                else:
                    pii_preview[key] = dictionary[key]

        # Add PII preview only if requested
        if show_pii_preview:
            dictionary['pii_preview'] = pii_preview
        
        dictionary['pii_summary'] = {
            'total_items': total_pii_found,
            'categories_found': pii_categories_found,
            'sensitive_items': sum(len(v) for k, v in dictionary.items() if k in ['Email', 'Phone', 'Personal Cell', 'Business Phone', 'SSN', 'Credit Card'] and v),
            'non_sensitive_items': sum(len(v) for k, v in dictionary.items() if k in ['Name', 'Location', 'Employer', 'Education', 'Occupation', 'Gender'] and v),
            'categories_breakdown': {
                'names': len(dictionary.get('Name', [])),
                'emails': len(dictionary.get('Email', [])),
                'phones': len(dictionary.get('Phone', [])) + len(dictionary.get('Personal Cell', [])),
                'addresses': len(dictionary.get('Address', [])),
                'locations': len(dictionary.get('Location', [])),
                'employers': len(dictionary.get('Employer', [])),
                'education': len(dictionary.get('Education', [])),
                'social_media': len(dictionary.get('Facebook Account', [])) + len(dictionary.get('Twitter Account', [])) + len(dictionary.get('Instagram Account', [])) + len(dictionary.get('LinkedIn Account', [])),
                'sensitive_docs': len(dictionary.get('SSN', [])) + len(dictionary.get('Credit Card', [])) + len(dictionary.get('Passport', [])) + len(dictionary.get('DDL', []))
            }
        }
        
        # Add debug mode data if requested
        if debug_mode:
            # Show full PII data in debug mode (WARNING: Contains sensitive data)
            debug_pii_data = {k: v for k, v in dictionary.items() if k in ['Name', 'Location', 'Email', 'Phone', 'Address', 'Employer', 'Education', 'DOB', 'Gender', 'Personal Cell', 'Business Phone', 'SSN', 'Credit Card'] and v}
            dictionary['debug_pii_data'] = debug_pii_data
            logger.warning(f"DEBUG MODE - Full PII data exposed: {debug_pii_data}")
        else:
            # In non-debug mode, ensure no sensitive data is exposed
            dictionary['debug_pii_data'] = "Debug mode disabled - sensitive data not exposed"

        # Log completion stats
        execution_time = time.time() - execution_start
        logger.info(f"OPTIMIZED extraction completed in {execution_time:.2f} seconds")
        logger.info(f"Overall Risk Score: {overall_risk_score:.2f} ({risk_level})")
        logger.info(f"Total PII found: {total_pii_found} items across {pii_categories_found} categories")
        logger.info(f"Performance improvement: Concurrent processing enabled")
        
        # Debug: Log final response structure
        logger.info(f"Final response dictionary keys: {list(dictionary.keys())}")
        
        # Show actual PII values (masked for security)
        pii_preview = {k: v for k, v in dictionary.items() if k in ['Name', 'Location', 'Email', 'Phone', 'Address', 'Location', 'Employer', 'Education', 'DOB', 'Gender'] and v}
        masked_pii = mask_sensitive_pii(pii_preview)
        logger.info(f"PII data in response (masked): {masked_pii}")
        
        # Show non-sensitive PII in full
        non_sensitive_pii = {k: v for k, v in dictionary.items() if k in ['Name', 'Location', 'Employer', 'Education', 'Occupation', 'Gender'] and v}
        logger.info(f"Non-sensitive PII: {non_sensitive_pii}")
        
        # Show counts for all PII categories
        pii_counts = {k: len(v) if isinstance(v, list) else 0 for k, v in dictionary.items() if k in ['Name', 'Location', 'Email', 'Phone', 'DOB', 'Address', 'Gender', 'Employer', 'Education', 'Birth Place', 'Personal Cell', 'Business Phone', 'Facebook Account', 'Twitter Account', 'Instagram Account', 'LinkedIn Account', 'TikTok Account', 'YouTube Account', 'DDL', 'Passport', 'Credit Card', 'SSN', 'Family Members', 'Occupation', 'Salary', 'Website'] and v}
        logger.info(f"PII counts by category: {pii_counts}")

        # Log exact JSON response being sent to frontend
        logger.info("="*100)
        logger.info("EXACT JSON RESPONSE BEING SENT TO FRONTEND")
        logger.info("="*100)

        import json
        try:
            # Convert to JSON string
            json_response = json.dumps(dictionary, indent=2)
            
            # Log the exact JSON
            logger.info("RAW JSON RESPONSE:")
            logger.info(json_response)
            
            # Log response size
            response_size = len(json_response)
            logger.info(f"\nResponse size: {response_size:,} characters ({response_size/1024:.2f} KB)")
            
            logger.info("="*100)
            logger.info("END OF JSON RESPONSE")
            logger.info("="*100)

        except Exception as e:
            logger.error(f"Error converting to JSON: {e}")
            logger.info(f"Raw dictionary: {dictionary}")

        return dictionary  # Return the flat dictionary like the original Flask version

    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(f"Extraction error: {e}")
        return {"error": f"Extraction failed: {str(e)}"}
