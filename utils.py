import re
import logging
from urllib.parse import urlparse, parse_qs
from difflib import SequenceMatcher
from typing import List, Dict

logger = logging.getLogger(__name__)

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
