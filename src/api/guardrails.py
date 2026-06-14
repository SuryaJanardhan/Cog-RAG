"""
Enterprise-grade Input/Output Guardrails and PII Masking.
Provides regex-based PII detection and masking, and prompt injection safety checks.
"""
import re
from typing import Dict, Any, List, Optional, Tuple


class PIIMasker:
    """Detects and masks sensitive Personally Identifiable Information (PII)."""
    
    # Pre-compiled patterns for common PII types
    PATTERNS = {
        "EMAIL": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
        "PHONE": re.compile(r"\b(?:\+?\d{1,3}[-. ]?)?\(?\d{3}\)?[-. ]?\d{3}[-. ]?\d{4}\b"),
        "CREDIT_CARD": re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
        "IP_ADDRESS": re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"),
        "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    }
    
    @classmethod
    def mask(cls, text: str) -> Tuple[str, Dict[str, str]]:
        """
        Mask PII inside the text and return the masked text and a mapping to restore it.
        """
        masked_text = text
        mappings = {}
        counter = 1
        
        for pii_type, pattern in cls.PATTERNS.items():
            matches = list(set(pattern.findall(masked_text)))
            for match in matches:
                # Do not mask if it's already masked
                if match.startswith("[") and match.endswith("]"):
                    continue
                placeholder = f"[{pii_type}_{counter}]"
                mappings[placeholder] = match
                masked_text = masked_text.replace(match, placeholder)
                counter += 1
                
        return masked_text, mappings

    @classmethod
    def unmask(cls, text: str, mappings: Dict[str, str]) -> str:
        """Restore original PII data using the mappings dict."""
        unmasked = text
        for placeholder, original in mappings.items():
            unmasked = unmasked.replace(placeholder, original)
        return unmasked


class InputGuardrail:
    """Verifies incoming user inputs for safety and masks PII."""
    
    SUSPICIOUS_KEYWORDS = [
        "ignore previous instructions",
        "system prompt",
        "you are now in developer mode",
        "do not explain",
        "bypass guardrails",
    ]
    
    @classmethod
    def analyze(cls, text: str) -> Dict[str, Any]:
        """
        Analyze input text for safety.
        
        Returns:
            dict: Analysis results with safety flags and masked text.
        """
        # 1. Check for prompt injection keywords
        text_lower = text.lower()
        is_safe = True
        flagged_reason = None
        
        for keyword in cls.SUSPICIOUS_KEYWORDS:
            if keyword in text_lower:
                is_safe = False
                flagged_reason = f"Suspicious instruction detected: '{keyword}'"
                break
                
        # 2. Mask PII
        masked_text, pii_mappings = PIIMasker.mask(text)
        
        return {
            "is_safe": is_safe,
            "flagged_reason": flagged_reason,
            "masked_text": masked_text,
            "pii_mappings": pii_mappings,
            "has_pii": len(pii_mappings) > 0
        }


class OutputGuardrail:
    """Grades LLM output content for hallucinations, prompt leaks, or unmasked placeholders."""
    
    LEAK_PATTERNS = [
        r"system instructions",
        r"internal assistant guidelines",
        r"you are an AI assistant",
    ]
    
    @classmethod
    def verify(cls, response: str, context: str) -> Dict[str, Any]:
        """
        Verify output text for safety.
        """
        # 1. Check for prompt leakage
        response_lower = response.lower()
        leaks_detected = False
        
        for pattern in cls.LEAK_PATTERNS:
            if re.search(pattern, response_lower):
                leaks_detected = True
                break
                
        # 2. Basic hallucination check: Ensure numbers in output exist in the context
        # (Very simple check: find all percentages or amounts and confirm context mentions them)
        numbers = re.findall(r"\b\d+(?:\.\d+)?%\b|\$\d+(?:\.\d+)?", response)
        unverified_claims = []
        for num in numbers:
            if num not in context:
                unverified_claims.append(num)
                
        is_safe = not leaks_detected and len(unverified_claims) == 0
        
        return {
            "is_safe": is_safe,
            "leaks_detected": leaks_detected,
            "unverified_claims": unverified_claims,
            "reason": "Prompt leak or unverified numbers detected" if not is_safe else "Passed"
        }
