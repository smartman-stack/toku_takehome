"""
Data loading and parsing module for TokuTel RAG system.
Handles loading and structuring all data sources with citation tracking.
Implements intelligent chunking for optimal retrieval.
"""
import csv
import json
import yaml
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class Document:
    """Represents a document chunk with citation information."""
    content: str
    source_file: str
    citation: str  # Format: [source_file#identifier]
    metadata: Dict[str, Any] = field(default_factory=dict)
    doc_type: str = "general"  # plan, policy, transcript, faq, feature
    keywords: List[str] = field(default_factory=list)  # For BM25 boosting
    
    def __hash__(self):
        return hash(self.citation)
    
    def __eq__(self, other):
        if isinstance(other, Document):
            return self.citation == other.citation
        return False


class DataLoader:
    """Loads and structures data from all sources with intelligent chunking."""
    
    def __init__(self):
        self.documents: List[Document] = []
        self._plans_data: Dict[str, Dict] = {}  # Cache for plan lookups
        self._features_data: Dict[str, List[str]] = {}  # Cache for feature lookups
    
    def load_plans_csv(self, filepath: str = "data/plans.csv") -> List[Document]:
        """Load plans.csv and create documents with row citations."""
        docs = []
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader, start=2):  # Start at 2 (row 1 is header)
                plan_id = row['plan_id']
                
                # Cache plan data for lookups
                self._plans_data[plan_id] = row
                
                # Create comprehensive, searchable content
                content_parts = [
                    f"Plan: {row['name']}",
                    f"Plan ID: {plan_id}",
                    f"Region: {row['region']}",
                    f"Minutes included: {row['minutes']}",
                    f"SMS included: {row['sms']}",
                    f"Price: ${row['price_usd']} USD per month",
                ]
                if row.get('notes'):
                    content_parts.append(f"Features/Notes: {row['notes']}")
                
                content = ". ".join(content_parts) + "."
                citation = f"[plans.csv#row={idx}]"
                
                # Extract keywords for BM25
                keywords = [
                    row['name'].lower(),
                    plan_id.lower(),
                    row['region'].lower(),
                ]
                if row.get('notes'):
                    # Extract key terms from notes
                    notes_lower = row['notes'].lower()
                    if 'call recording' in notes_lower:
                        keywords.extend(['call recording', 'recording'])
                    if 'sso' in notes_lower:
                        keywords.append('sso')
                    if 'whatsapp' in notes_lower:
                        keywords.extend(['whatsapp', 'wa', 'api'])
                    if 'sentiment' in notes_lower:
                        keywords.append('sentiment analysis')
                
                docs.append(Document(
                    content=content,
                    source_file="plans.csv",
                    citation=citation,
                    metadata={
                        "plan_id": plan_id,
                        "plan_name": row['name'],
                        "row": idx,
                        "region": row['region'],
                        "price_usd": float(row['price_usd']),
                        "minutes": int(row['minutes']),
                        "sms": int(row['sms']),
                        "notes": row.get('notes', '')
                    },
                    doc_type="plan",
                    keywords=keywords
                ))
        return docs
    
    def load_kb_yaml(self, filepath: str = "data/kb.yaml") -> List[Document]:
        """Load kb.yaml and create granular documents with section citations."""
        docs = []
        with open(filepath, 'r', encoding='utf-8') as f:
            kb_data = yaml.safe_load(f)
        
        # Company policies - create separate documents for each policy
        if 'company' in kb_data:
            company = kb_data['company']
            
            # PII Policy - important for compliance
            if 'pii_policy' in company:
                content = "PII Policy: " + " ".join(company['pii_policy'])
                docs.append(Document(
                    content=content,
                    source_file="kb.yaml",
                    citation="[kb.yaml#pii_policy]",
                    metadata={"section": "pii_policy", "type": "policy"},
                    doc_type="policy",
                    keywords=['pii', 'privacy', 'card', 'national id', 'phone', 'mask', 'security']
                ))
            
            # Escalation Policy - create separate docs for each level
            if 'escalation_policy' in company:
                escalation = company['escalation_policy']
                
                # Combined escalation document
                content_parts = ["Escalation Policy:"]
                for level, desc in escalation.items():
                    content_parts.append(f"{level.upper()}: {desc}")
                content = " ".join(content_parts)
                docs.append(Document(
                    content=content,
                    source_file="kb.yaml",
                    citation="[kb.yaml#escalation_policy]",
                    metadata={"section": "escalation_policy", "type": "policy", "levels": list(escalation.keys())},
                    doc_type="policy",
                    keywords=['escalation', 'escalate', 'p0', 'p1', 'p2', 'outage', 'incident', 'urgent', 'critical']
                ))
            
            # SLA - separate document for response times
            if 'sla' in company:
                sla = company['sla']
                standard_hours = sla.get('standard_response_hours', 24)
                enterprise_hours = sla.get('enterprise_response_hours', 4)
                content = f"SLA Response Times: Standard customers receive response within {standard_hours} hours. Enterprise customers receive priority response within {enterprise_hours} hours."
                docs.append(Document(
                    content=content,
                    source_file="kb.yaml",
                    citation="[kb.yaml#sla]",
                    metadata={
                        "section": "sla",
                        "type": "policy",
                        "standard_hours": standard_hours,
                        "enterprise_hours": enterprise_hours
                    },
                    doc_type="policy",
                    keywords=['sla', 'response time', 'support', 'hours', 'enterprise', 'standard']
                ))
        
        # Pricing rules
        if 'pricing_rules' in kb_data:
            pricing = kb_data['pricing_rules']
            
            # Discounts - create rich, queryable content
            if 'discounts' in pricing:
                discounts = pricing['discounts']
                discount_parts = []
                for discount in discounts:
                    condition = discount['condition'].replace('_', ' ')
                    value = discount['value_pct']
                    discount_parts.append(f"{condition}: {value}% discount")
                
                content = "Available Discounts: " + ". ".join(discount_parts) + ". Discounts may be combined - contact sales for exact pricing with multiple discounts."
                docs.append(Document(
                    content=content,
                    source_file="kb.yaml",
                    citation="[kb.yaml#discounts]",
                    metadata={
                        "section": "discounts",
                        "type": "pricing",
                        "discounts": discounts
                    },
                    doc_type="policy",
                    keywords=['discount', 'annual', 'prepay', 'nonprofit', 'pricing', 'savings', 'combine']
                ))
            
            # WhatsApp API quota tiers - detailed document
            if 'wa_api_quota_tiers' in pricing:
                tiers = pricing['wa_api_quota_tiers']
                tier_parts = []
                for tier_name, quota in tiers.items():
                    tier_parts.append(f"{tier_name.upper()}: {quota}")
                
                content = f"WhatsApp API Quota Tiers: {'. '.join(tier_parts)}. Higher tiers available as add-on - contact sales to upgrade quota."
                docs.append(Document(
                    content=content,
                    source_file="kb.yaml",
                    citation="[kb.yaml#wa_api_quota_tiers]",
                    metadata={
                        "section": "wa_api_quota_tiers",
                        "type": "pricing",
                        "tiers": tiers
                    },
                    doc_type="policy",
                    keywords=['whatsapp', 'wa', 'api', 'quota', 'tier', 'conversations', 'limit', 'exceeded']
                ))
        
        # Features matrix - create individual feature documents
        if 'features_matrix' in kb_data:
            features = kb_data['features_matrix']
            self._features_data = features
            
            # Combined features document
            feature_parts = []
            for feature, plans in features.items():
                feature_name = feature.replace('_', ' ')
                feature_parts.append(f"{feature_name}: available on {', '.join(plans)}")
            
            content = "Features Matrix: " + ". ".join(feature_parts) + "."
            docs.append(Document(
                content=content,
                source_file="kb.yaml",
                citation="[kb.yaml#features_matrix]",
                metadata={"section": "features_matrix", "type": "features", "features": features},
                doc_type="feature",
                keywords=['feature', 'call recording', 'sentiment', 'whatsapp', 'sso', 'available', 'include']
            ))
            
            # Individual feature documents for better retrieval
            for feature, plans in features.items():
                feature_name = feature.replace('_', ' ')
                plan_names = [self._get_plan_name(p) for p in plans]
                content = f"Feature: {feature_name}. This feature is available on: {', '.join(plan_names)}. Plan IDs: {', '.join(plans)}."
                docs.append(Document(
                    content=content,
                    source_file="kb.yaml",
                    citation=f"[kb.yaml#features_matrix#{feature}]",
                    metadata={"section": "features_matrix", "feature": feature, "plans": plans},
                    doc_type="feature",
                    keywords=[feature_name, feature.replace('_', ''), *[p.lower() for p in plans]]
                ))
        
        return docs
    
    def _get_plan_name(self, plan_id: str) -> str:
        """Get plan name from plan ID."""
        if plan_id in self._plans_data:
            return self._plans_data[plan_id].get('name', plan_id)
        # Fallback: convert plan ID to readable name
        return plan_id.replace('-', ' ').title()
    
    def load_transcripts_json(self, filepath: str = "data/transcripts.json") -> List[Document]:
        """Load transcripts.json and create documents with transcript citations."""
        docs = []
        with open(filepath, 'r', encoding='utf-8') as f:
            transcripts = json.load(f)
        
        for transcript in transcripts:
            transcript_id = transcript['id']
            channel = transcript.get('channel', 'unknown')
            region = transcript.get('region', 'unknown')
            customer = transcript.get('customer', 'unknown')
            text = transcript['text']
            
            # Parse the transcript to extract question and answer
            q_and_a = self._parse_transcript(text)
            
            # Create searchable content
            content = f"Customer Support Example ({channel} channel, {region} region, customer: {customer}): {text}"
            citation = f"[transcripts.json#{transcript_id}]"
            
            # Extract keywords from transcript
            keywords = []
            text_lower = text.lower()
            if 'call recording' in text_lower or 'recording' in text_lower:
                keywords.extend(['call recording', 'recording'])
            if 'sso' in text_lower:
                keywords.append('sso')
            if 'whatsapp' in text_lower or 'quota' in text_lower:
                keywords.extend(['whatsapp', 'quota', 'exceeded'])
            if 'upgrade' in text_lower:
                keywords.append('upgrade')
            
            docs.append(Document(
                content=content,
                source_file="transcripts.json",
                citation=citation,
                metadata={
                    "transcript_id": transcript_id,
                    "channel": channel,
                    "region": region,
                    "customer": customer,
                    "parsed": q_and_a
                },
                doc_type="transcript",
                keywords=keywords
            ))
        
        return docs
    
    def _parse_transcript(self, text: str) -> Dict[str, str]:
        """Parse transcript text to extract caller/user question and agent response."""
        result = {"question": "", "answer": ""}
        
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith(('Caller:', 'User:', 'Customer:')):
                result["question"] = line.split(':', 1)[1].strip() if ':' in line else line
            elif line.startswith('Agent:'):
                result["answer"] = line.split(':', 1)[1].strip() if ':' in line else line
        
        return result
    
    def load_faq_jsonl(self, filepath: str = "data/faq.jsonl") -> List[Document]:
        """Load faq.jsonl and create documents."""
        docs = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f, start=1):
                faq = json.loads(line.strip())
                question = faq['q']
                answer = faq['a']
                
                # Create Q&A format content
                content = f"FAQ - Question: {question} Answer: {answer}"
                citation = f"[faq.jsonl#entry={idx}]"
                
                # Extract keywords
                keywords = []
                combined = (question + " " + answer).lower()
                if 'call recording' in combined or 'recording' in combined:
                    keywords.extend(['call recording', 'recording', 'lite', 'pro'])
                if 'whatsapp' in combined:
                    keywords.extend(['whatsapp', 'api', 'quota'])
                if 'sso' in combined:
                    keywords.append('sso')
                if 'discount' in combined:
                    keywords.extend(['discount', 'annual', 'prepay'])
                
                docs.append(Document(
                    content=content,
                    source_file="faq.jsonl",
                    citation=citation,
                    metadata={
                        "faq_id": idx,
                        "question": question,
                        "answer": answer
                    },
                    doc_type="faq",
                    keywords=keywords
                ))
        
        return docs
    
    def load_all(self) -> List[Document]:
        """Load all data sources and return combined documents."""
        self.documents = []
        
        # Load in order: plans first (for name lookups), then KB, then others
        self.documents.extend(self.load_plans_csv())
        self.documents.extend(self.load_kb_yaml())
        self.documents.extend(self.load_transcripts_json())
        self.documents.extend(self.load_faq_jsonl())
        
        return self.documents
    
    def get_plan_by_id(self, plan_id: str) -> Optional[Dict]:
        """Get plan data by ID."""
        return self._plans_data.get(plan_id)
    
    def get_plans_with_feature(self, feature: str) -> List[str]:
        """Get list of plan IDs that have a specific feature."""
        return self._features_data.get(feature, [])
    
    def get_all_plans(self) -> Dict[str, Dict]:
        """Get all plans data."""
        return self._plans_data.copy()
