"""
Context Reduction Utilities

Reduces the size of large context strings to prevent token limit issues.
"""

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


def truncate_text(text: str, max_chars: int = 2000, suffix: str = "...") -> str:
    """
    Truncate text to a maximum character count.
    
    Args:
        text: Text to truncate
        max_chars: Maximum characters (default: 2000)
        suffix: Suffix to add when truncated (default: "...")
        
    Returns:
        Truncated text
    """
    if not text or len(text) <= max_chars:
        return text
    
    return text[:max_chars - len(suffix)] + suffix


def summarize_statistical_profile(profile: str, max_length: int = 1500) -> str:
    """
    Summarize a statistical profile by extracting key metrics only.
    
    Args:
        profile: Full statistical profile text
        max_length: Maximum length for summary (default: 1500 chars)
        
    Returns:
        Summarized profile with key metrics only
    """
    if not profile or len(profile) <= max_length:
        return profile
    
    # Extract key sections
    key_sections = []
    
    # Extract address and coordinates
    address_match = re.search(r'ADDRESS:\s*(.+?)(?:\n|$)', profile)
    if address_match:
        key_sections.append(f"Address: {address_match.group(1).strip()}")
    
    # Extract 5-mile radius summary (most important)
    five_mile_match = re.search(
        r'===.*5.*MILE.*RADIUS.*?===(.*?)(?===|KEY MARKET|END OF)', 
        profile, 
        re.DOTALL | re.IGNORECASE
    )
    if five_mile_match:
        five_mile_text = five_mile_match.group(1)
        # Extract key metrics from 5-mile section
        metrics = []
        for pattern in [
            r'Total Population:\s*([^\n]+)',
            r'Median Household Income:\s*\$?([^\n]+)',
            r'Total Households:\s*([^\n]+)',
            r'Unemployment Rate:\s*([^\n]+)',
        ]:
            match = re.search(pattern, five_mile_text, re.IGNORECASE)
            if match:
                metrics.append(match.group(0).strip())
        
        if metrics:
            key_sections.append("5-Mile Radius Key Metrics:")
            key_sections.extend(metrics)
    
    # Extract KEY MARKET INSIGHTS section
    insights_match = re.search(
        r'=== KEY MARKET INSIGHTS ===(.*?)(?===|END OF)', 
        profile, 
        re.DOTALL | re.IGNORECASE
    )
    if insights_match:
        insights_text = insights_match.group(1)
        # Extract key points
        key_points = []
        for line in insights_text.split('\n'):
            line = line.strip()
            if line and (line.startswith('-') or ':' in line):
                key_points.append(line)
                if len(key_points) >= 5:  # Limit to top 5 insights
                    break
        
        if key_points:
            key_sections.append("\nKey Market Insights:")
            key_sections.extend(key_points[:5])
    
    # Combine sections
    summary = "\n\n".join(key_sections)
    
    # If still too long, truncate
    if len(summary) > max_length:
        summary = truncate_text(summary, max_length)
    
    return summary or truncate_text(profile, max_length)


def summarize_tapestry_insights(insights: str, max_length: int = 1000) -> str:
    """
    Summarize tapestry insights by extracting key information.
    
    Args:
        insights: Full tapestry insights text
        max_length: Maximum length for summary (default: 1000 chars)
        
    Returns:
        Summarized insights
    """
    if not insights or len(insights) <= max_length:
        return insights
    
    # Try to extract key paragraphs (first 2-3 paragraphs often contain summary)
    paragraphs = insights.split('\n\n')
    
    if len(paragraphs) >= 2:
        # Take first 2-3 paragraphs which usually contain overview
        summary = '\n\n'.join(paragraphs[:3])
        
        # If still too long, truncate
        if len(summary) > max_length:
            summary = truncate_text(summary, max_length)
        
        return summary
    
    # Fallback: truncate
    return truncate_text(insights, max_length)


def reduce_context_size(context: dict, max_total_chars: int = 5000) -> dict:
    """
    Reduce the size of a context dictionary by truncating large string values.
    
    Args:
        context: Context dictionary
        max_total_chars: Maximum total characters across all string values
        
    Returns:
        Reduced context dictionary
    """
    if not context:
        return context
    
    # Calculate current size
    total_chars = sum(
        len(str(v)) for v in context.values() if isinstance(v, str)
    )
    
    if total_chars <= max_total_chars:
        return context
    
    # Create reduced copy
    reduced = context.copy()
    
    # Priority order: reduce less important fields first
    reduction_priority = [
        'statistical_profile',  # Can be very large
        'tapestry_insights',     # Can be large
        'additional_context',    # Less critical
        'recent_analyses',       # Historical data
    ]
    
    # Reduce fields in priority order
    for field in reduction_priority:
        if field in reduced and isinstance(reduced[field], str):
            if field == 'statistical_profile':
                reduced[field] = summarize_statistical_profile(
                    reduced[field], 
                    max_length=1500
                )
            elif field == 'tapestry_insights':
                reduced[field] = summarize_tapestry_insights(
                    reduced[field],
                    max_length=1000
                )
            else:
                reduced[field] = truncate_text(reduced[field], max_length=500)
            
            # Check if we're under limit now
            total_chars = sum(
                len(str(v)) for v in reduced.values() if isinstance(v, str)
            )
            if total_chars <= max_total_chars:
                break
    
    logger.debug(
        f"Reduced context size from {total_chars:,} to "
        f"{sum(len(str(v)) for v in reduced.values() if isinstance(v, str)):,} chars"
    )
    
    return reduced

