"""
Policy enforcement module for PII masking, escalation, and SLA compliance.
"""
import re
from typing import Dict, Any, List


class PolicyEnforcer:
    """Enforces company policies on generated responses."""
    
    def __init__(self):
        # Patterns for PII detection
        self.card_pattern = re.compile(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b')
        self.phone_pattern = re.compile(r'\b\d{3,4}[\s-]?\d{3,4}[\s-]?\d{3,4}\b')
        # National ID pattern: alphanumeric with at least 2 digits (excludes plan IDs which are mostly letters)
        self.national_id_pattern = re.compile(r'\b(?=\w*\d\w*\d)[A-Z0-9]{6,12}\b')
        
        # Whitelist of known business terms that should NOT be masked
        self.whitelist = {
            'MERANTI', 'KAPOK', 'RAINTREE',  # Plan prefixes
            'CC', 'Lite', 'Pro', 'Plus', 'UC',  # Plan suffixes
            'SG', 'MY', 'ID', 'TH', 'PH',  # Region codes
            't1', 't2', 't3',  # Tier identifiers
        }
    
    def mask_pii(self, text: str) -> str:
        """
        Mask PII according to policy:
        - Never echo card numbers or national IDs
        - Mask phone numbers except last 3 digits
        """
        # Remove card numbers completely
        text = self.card_pattern.sub('[CARD NUMBER REDACTED]', text)
        
        # Remove national IDs completely (but exclude whitelisted terms)
        def should_mask_id(match):
            matched_text = match.group(0)
            # Check if any whitelisted term is in the matched text
            for term in self.whitelist:
                if term in matched_text:
                    return False
            return True
        
        def replace_id(match):
            if should_mask_id(match):
                return '[ID REDACTED]'
            return match.group(0)
        
        text = self.national_id_pattern.sub(replace_id, text)
        
        # Mask phone numbers (keep last 3 digits)
        def mask_phone(match):
            phone = match.group(0).replace(' ', '').replace('-', '')
            if len(phone) >= 3:
                return '*' * (len(phone) - 3) + phone[-3:]
            return match.group(0)
        
        text = self.phone_pattern.sub(mask_phone, text)
        
        return text
    
    def check_escalation_needed(self, query: str, response: str) -> Dict[str, Any]:
        """
        Determine if escalation is needed based on query content.
        Returns escalation level and instructions.
        """
        query_lower = query.lower()
        response_lower = response.lower()
        
        # P0: Outage or data-loss risk
        p0_keywords = ['outage', 'down', 'data loss', 'data-loss', 'critical', 'emergency', 'not working']
        if any(keyword in query_lower for keyword in p0_keywords):
            return {
                "level": "p0",
                "message": "P0 escalation required: Outage or data-loss risk — page on-call within 5 minutes.",
                "needed": True
            }
        
        # P1: Multiple customers blocked
        p1_keywords = ['multiple customers', 'many users', 'widespread', 'affecting multiple']
        if any(keyword in query_lower for keyword in p1_keywords):
            return {
                "level": "p1",
                "message": "P1 escalation required: Multiple customers blocked — escalate to Incident Manager within 30 minutes.",
                "needed": True
            }
        
        # P2: Single customer degradation
        p2_keywords = ['degraded', 'slow', 'issue', 'problem', 'error']
        if any(keyword in query_lower for keyword in p2_keywords):
            return {
                "level": "p2",
                "message": "P2 escalation: Single-customer degradation — open Jira and respond within 4 business hours.",
                "needed": True
            }
        
        return {"level": None, "message": "", "needed": False}
    
    def apply_sla_guidance(self, is_enterprise: bool = False) -> str:
        """
        Return SLA guidance based on customer type.
        """
        if is_enterprise:
            return "Response SLA: 4 business hours for enterprise customers."
        return "Response SLA: 24 hours for standard customers."
    
    def enforce_policies(self, text: str, query: str = "", is_enterprise: bool = False) -> Dict[str, Any]:
        """
        Apply all policy enforcement to text.
        Returns dict with masked text, escalation info, and SLA guidance.
        """
        masked_text = self.mask_pii(text)
        escalation = self.check_escalation_needed(query, masked_text)
        sla_guidance = self.apply_sla_guidance(is_enterprise)
        
        return {
            "text": masked_text,
            "escalation": escalation,
            "sla_guidance": sla_guidance
        }
