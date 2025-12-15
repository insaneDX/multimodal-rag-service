from typing import List, Dict, Set
import re

import src.logger_config
from loguru import logger


def verify_answer(answer: str, retrieved: List[Dict]) -> Dict:
    """
    Verify the LLM answer against retrieved evidence using keyword matching.
    
    Args:
        answer: The generated answer from the LLM.
        retrieved: Retrieved documents with metadata.

    Returns:
        dict: Verification report with status, issues, score, and notes.
    """
    if not answer or not answer.strip():
        return {
            "status": "flagged",
            "issues": ["Empty answer"],
            "score": 0.0,
            "notes": "The answer is empty or contains only whitespace."
        }
    
    normalized_answer = answer.lower()
    
    # Combine all evidence text
    evidence_texts = []
    for item in retrieved:
        doc = item.get("document", "")
        if doc:
            evidence_texts.append(doc.lower())
        
        # Also include caption for images
        caption = item.get("metadata", {}).get("caption", "")
        if caption:
            evidence_texts.append(caption.lower())
    
    combined_evidence = " ".join(evidence_texts)
    
    # Extract meaningful keywords from answer (excluding common words)
    stop_words = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
        'ought', 'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
        'from', 'as', 'into', 'through', 'during', 'before', 'after', 'above',
        'below', 'between', 'under', 'again', 'further', 'then', 'once', 'here',
        'there', 'when', 'where', 'why', 'how', 'all', 'each', 'few', 'more',
        'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
        'same', 'so', 'than', 'too', 'very', 'just', 'and', 'but', 'if', 'or',
        'because', 'until', 'while', 'this', 'that', 'these', 'those', 'what',
        'which', 'who', 'whom', 'whose', 'i', 'you', 'he', 'she', 'it', 'we',
        'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'its',
        'our', 'their', 'cannot', 'verify', 'provided', 'documents', 'based',
        'evidence', 'according', 'source'
    }
    
    # Extract words, filtering out short words, URLs, and stop words
    words = re.findall(r'\b[a-z]+\b', normalized_answer)
    keywords = [
        word for word in words
        if len(word) > 4 
        and word not in stop_words
        and not word.startswith('http')
    ]
    
    # Deduplicate keywords
    unique_keywords = list(set(keywords))
    
    if not unique_keywords:
        return {
            "status": "verified",
            "issues": [],
            "score": 1.0,
            "notes": "No significant keywords to verify (answer may be very short or generic)."
        }
    
    # Check which keywords are missing from evidence
    missing_keywords = [kw for kw in unique_keywords if kw not in combined_evidence]
    
    # Calculate verification score
    match_ratio = 1.0 - (len(missing_keywords) / len(unique_keywords))
    
    issues = []
    if missing_keywords:
        issues.append(f"{len(missing_keywords)}/{len(unique_keywords)} keywords not found in evidence")
    
    # Determine status based on match ratio
    if match_ratio >= 0.8:
        status = "verified"
        notes = f"High confidence: {match_ratio:.1%} of keywords found in evidence."
    elif match_ratio >= 0.5:
        status = "partial"
        notes = f"Moderate confidence: {match_ratio:.1%} of keywords found. Missing: {', '.join(missing_keywords[:5])}..."
    else:
        status = "flagged"
        notes = f"Low confidence: Only {match_ratio:.1%} of keywords found. Missing: {', '.join(missing_keywords[:10])}..."
    
    logger.debug("Verification: status={}, score={:.2f}, missing={}", 
                 status, match_ratio, len(missing_keywords))
    
    return {
        "status": status,
        "issues": issues,
        "score": match_ratio,
        "notes": notes
    }


def extract_citations(answer: str) -> List[str]:
    """Extract citation IDs from answer text."""
    return re.findall(r"\[([^\]]+)\]", answer)


def validate_citations(answer: str, retrieved: List[Dict]) -> Dict:
    """
    Validate that citations in the answer reference retrieved documents.
    
    Args:
        answer: The LLM answer containing citations
        retrieved: Retrieved documents
        
    Returns:
        Validation report
    """
    citations = extract_citations(answer)
    retrieved_ids = {r['id'] for r in retrieved}
    
    valid_citations = [c for c in citations if c in retrieved_ids]
    invalid_citations = [c for c in citations if c not in retrieved_ids]
    
    if not citations:
        return {
            "status": "no_citations",
            "valid": [],
            "invalid": [],
            "notes": "No citations found in answer"
        }
    
    if invalid_citations:
        return {
            "status": "invalid_citations",
            "valid": valid_citations,
            "invalid": invalid_citations,
            "notes": f"Found {len(invalid_citations)} citations referencing unknown sources"
        }
    
    return {
        "status": "valid",
        "valid": valid_citations,
        "invalid": [],
        "notes": f"All {len(valid_citations)} citations are valid"
    }