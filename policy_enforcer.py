"""
Policy enforcement module for PII masking, escalation, and SLA compliance.
Enhanced with configurable policies and better pattern matching.
"""
import re
from typing import Dict, Any, List, Optional, Set
from config import get_config, Config


class PolicyEnforcer:
    """Enforces company policies on generated responses."""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or get_config()
        
        # PII detection patterns
        self.patterns = {
            # Credit card: 16 digits with optional spaces/dashes
            'card': re.compile(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b'),
            
            # Phone numbers: Various formats
            'phone': re.compile(r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}\b'),
            
            # National ID: Alphanumeric 6-12 chars with at least 2 digits
            # Excludes plan IDs which are mostly letters with dashes
            'national_id': re.compile(r'\b(?=\w*\d\w*\d)[A-Z0-9]{6,12}\b'),
            
            # Email addresses
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
        }
        
        # Whitelist of business terms that should NOT be masked
        self.whitelist: Set[str] = {
            # Plan prefixes
            'MERANTI', 'KAPOK', 'RAINTREE',
            # Plan suffixes and types
            'CC', 'UC', 'Lite', 'Pro', 'Plus', 'Voice', 'Omni',
            # Region codes
            'SG', 'MY', 'ID', 'TH', 'PH', 'VN',
            # Tier identifiers
            'T1', 'T2', 'T3', 't1', 't2', 't3',
            # Common plan patterns
            'cc-lite', 'cc-pro', 'uc', 'voice-100', 'omni-Plus',
        }
        
        # Escalation keywords by priority level
        self.escalation_keywords = {
            'p0': {
                'keywords': ['outage', 'down', 'data loss', 'data-loss', 'critical', 
                            'emergency', 'completely broken', 'total failure', 'all users affected'],
                'message': 'P0 escalation required: Outage or data-loss risk — page on-call within 5 minutes.'
            },
            'p1': {
                'keywords': ['multiple customers', 'many users', 'widespread', 
                            'affecting multiple', 'several accounts', 'team affected'],
                'message': 'P1 escalation required: Multiple customers blocked — escalate to Incident Manager within 30 minutes.'
            },
            'p2': {
                'keywords': ['degraded', 'slow', 'issue', 'problem', 'error', 
                            'not working properly', 'intermittent', 'sometimes fails'],
                'message': 'P2 escalation: Single-customer degradation — open Jira and respond within 4 business hours.'
            }
        }
    
    def _should_mask(self, text: str, pattern_name: str) -> bool:
        """Check if text should be masked (not in whitelist)."""
        # Check against whitelist
        for term in self.whitelist:
            if term.lower() in text.lower():
                return False
            # Also check if text is part of a plan ID pattern
            if re.match(r'^[A-Z]+-[a-z]+-[a-z]+$', text, re.IGNORECASE):
                return False
        return True
    
    def mask_pii(self, text: str) -> str:
        """
        Mask PII according to company policy:
        - Never echo card numbers back to customers
        - Never echo national IDs
        - Mask phone numbers except last 3 digits
        - Mask email addresses
        """
        if not self.config.policy.mask_pii:
            return text
        
        # Mask card numbers completely
        text = self.patterns['card'].sub('[CARD REDACTED]', text)
        
        # Mask national IDs (with whitelist check)
        def replace_national_id(match):
            matched = match.group(0)
            if self._should_mask(matched, 'national_id'):
                return '[ID REDACTED]'
            return matched
        
        text = self.patterns['national_id'].sub(replace_national_id, text)
        
        # Mask phone numbers (keep last 3 digits for verification)
        def mask_phone(match):
            phone = match.group(0)
            # Don't mask if it looks like a plan feature (e.g., "100 minutes")
            if any(unit in phone for unit in ['min', 'sms', 'conv']):
                return phone
            # Clean phone and mask
            cleaned = re.sub(r'[\s\-\(\)]', '', phone)
            if len(cleaned) >= 7:  # Only mask if it's phone-length
                return '*' * (len(cleaned) - 3) + cleaned[-3:]
            return phone
        
        text = self.patterns['phone'].sub(mask_phone, text)
        
        # Mask emails
        def mask_email(match):
            email = match.group(0)
            parts = email.split('@')
            if len(parts) == 2:
                username = parts[0]
                domain = parts[1]
                masked_user = username[0] + '***' if len(username) > 1 else '***'
                return f"{masked_user}@{domain}"
            return '[EMAIL REDACTED]'
        
        text = self.patterns['email'].sub(mask_email, text)
        
        return text
    
    def check_escalation_needed(self, query: str, response: str = "") -> Dict[str, Any]:
        """
        Determine if escalation is needed based on query content.
        Returns escalation level and instructions.
        
        Checks in order of severity: P0 -> P1 -> P2
        """
        if not self.config.policy.enable_escalation_detection:
            return {"level": None, "message": "", "needed": False}
        
        query_lower = query.lower()
        response_lower = response.lower()
        combined = query_lower + " " + response_lower
        
        # Check each escalation level in priority order
        for level in ['p0', 'p1', 'p2']:
            keywords = self.escalation_keywords[level]['keywords']
            if any(keyword in combined for keyword in keywords):
                return {
                    "level": level,
                    "message": self.escalation_keywords[level]['message'],
                    "needed": True
                }
        
        return {"level": None, "message": "", "needed": False}
    
    def get_sla_guidance(self, is_enterprise: bool = False) -> Dict[str, Any]:
        """
        Return SLA guidance based on customer type.
        """
        if is_enterprise:
            return {
                "type": "enterprise",
                "response_hours": 4,
                "message": "Response SLA: 4 business hours for enterprise customers.",
                "priority": "high"
            }
        return {
            "type": "standard",
            "response_hours": 24,
            "message": "Response SLA: 24 hours for standard customers.",
            "priority": "normal"
        }
    
    def enforce_policies(self, text: str, query: str = "", 
                        is_enterprise: bool = False) -> Dict[str, Any]:
        """
        Apply all policy enforcement to text.
        
        Args:
            text: The response text to enforce policies on
            query: The original query (for escalation detection)
            is_enterprise: Whether the customer is enterprise tier
        
        Returns:
            Dict with:
            - text: Policy-compliant text
            - escalation: Escalation info
            - sla_guidance: SLA info
            - pii_masked: Whether PII was masked
        """
        original_text = text
        
        # Apply PII masking
        masked_text = self.mask_pii(text)
        pii_was_masked = masked_text != original_text
        
        # Check escalation
        escalation = self.check_escalation_needed(query, masked_text)
        
        # Get SLA guidance
        is_ent = is_enterprise or self.config.policy.default_customer_type == "enterprise"
        sla = self.get_sla_guidance(is_ent)
        
        return {
            "text": masked_text,
            "escalation": escalation,
            "sla_guidance": sla["message"],
            "sla_details": sla,
            "pii_masked": pii_was_masked
        }
    
    def validate_response(self, response: str, context_docs: List[Any]) -> Dict[str, Any]:
        """
        Validate that response is properly grounded in context.
        
        Args:
            response: Generated response
            context_docs: Documents used as context
        
        Returns:
            Validation result with warnings if any
        """
        warnings = []
        
        # Check for potential hallucinations (prices/features not in context)
        price_pattern = re.compile(r'\$[\d,]+(?:\.\d{2})?')
        response_prices = set(price_pattern.findall(response))
        
        context_text = " ".join([doc.content if hasattr(doc, 'content') else str(doc) 
                                  for doc in context_docs])
        context_prices = set(price_pattern.findall(context_text))
        
        ungrounded_prices = response_prices - context_prices
        if ungrounded_prices:
            warnings.append(f"Potentially ungrounded prices: {ungrounded_prices}")
        
        # Check for proper citation format
        citation_pattern = re.compile(r'\[[^\]]+#[^\]]+\]')
        citations = citation_pattern.findall(response)
        
        return {
            "valid": len(warnings) == 0,
            "warnings": warnings,
            "citation_count": len(citations)
        }
