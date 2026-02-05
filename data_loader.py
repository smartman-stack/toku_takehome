"""
Data loading and parsing module for TokuTel RAG system.
Handles loading and structuring all data sources with citation tracking.
"""
import csv
import json
import yaml
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class Document:
    """Represents a document chunk with citation information."""
    content: str
    source_file: str
    citation: str  # Format: [source_file#identifier]
    metadata: Dict[str, Any]


class DataLoader:
    """Loads and structures data from all sources."""
    
    def __init__(self):
        self.documents: List[Document] = []
    
    def load_plans_csv(self, filepath: str = "data/plans.csv") -> List[Document]:
        """Load plans.csv and create documents with row citations."""
        docs = []
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader, start=2):  # Start at 2 (row 1 is header)
                # Create a comprehensive description of the plan
                content_parts = [
                    f"Plan: {row['name']} (ID: {row['plan_id']})",
                    f"Region: {row['region']}",
                    f"Minutes: {row['minutes']}",
                    f"SMS: {row['sms']}",
                    f"Price: ${row['price_usd']} USD",
                ]
                if row.get('notes'):
                    content_parts.append(f"Notes: {row['notes']}")
                
                content = ". ".join(content_parts)
                citation = f"[plans.csv#row={idx}]"
                
                docs.append(Document(
                    content=content,
                    source_file="plans.csv",
                    citation=citation,
                    metadata={"plan_id": row['plan_id'], "row": idx, "region": row['region']}
                ))
        return docs
    
    def load_kb_yaml(self, filepath: str = "data/kb.yaml") -> List[Document]:
        """Load kb.yaml and create documents with section citations."""
        docs = []
        with open(filepath, 'r', encoding='utf-8') as f:
            kb_data = yaml.safe_load(f)
        
        # Company policies
        if 'company' in kb_data:
            company = kb_data['company']
            
            # PII Policy
            if 'pii_policy' in company:
                content = "PII Policy: " + ". ".join(company['pii_policy'])
                docs.append(Document(
                    content=content,
                    source_file="kb.yaml",
                    citation="[kb.yaml#pii_policy]",
                    metadata={"section": "pii_policy", "type": "policy"}
                ))
            
            # Escalation Policy
            if 'escalation_policy' in company:
                escalation = company['escalation_policy']
                content_parts = []
                for level, desc in escalation.items():
                    content_parts.append(f"{level.upper()}: {desc}")
                content = "Escalation Policy: " + ". ".join(content_parts)
                docs.append(Document(
                    content=content,
                    source_file="kb.yaml",
                    citation="[kb.yaml#escalation_policy]",
                    metadata={"section": "escalation_policy", "type": "policy"}
                ))
            
            # SLA
            if 'sla' in company:
                sla = company['sla']
                content = f"SLA: Standard response {sla.get('standard_response_hours', 'N/A')} hours. Enterprise response {sla.get('enterprise_response_hours', 'N/A')} hours."
                docs.append(Document(
                    content=content,
                    source_file="kb.yaml",
                    citation="[kb.yaml#sla]",
                    metadata={"section": "sla", "type": "policy"}
                ))
        
        # Pricing rules
        if 'pricing_rules' in kb_data:
            pricing = kb_data['pricing_rules']
            
            # Discounts
            if 'discounts' in pricing:
                discount_parts = []
                for discount in pricing['discounts']:
                    discount_parts.append(f"{discount['condition']}: {discount['value_pct']}% discount")
                content = "Discount Rules: " + ". ".join(discount_parts)
                docs.append(Document(
                    content=content,
                    source_file="kb.yaml",
                    citation="[kb.yaml#discounts]",
                    metadata={"section": "discounts", "type": "pricing"}
                ))
            
            # WhatsApp API quota tiers
            if 'wa_api_quota_tiers' in pricing:
                tiers = pricing['wa_api_quota_tiers']
                tier_parts = [f"{k}: {v}" for k, v in tiers.items()]
                content = "WhatsApp API Quota Tiers: " + ". ".join(tier_parts)
                docs.append(Document(
                    content=content,
                    source_file="kb.yaml",
                    citation="[kb.yaml#wa_api_quota_tiers]",
                    metadata={"section": "wa_api_quota_tiers", "type": "pricing"}
                ))
        
        # Features matrix
        if 'features_matrix' in kb_data:
            features = kb_data['features_matrix']
            feature_parts = []
            for feature, plans in features.items():
                feature_parts.append(f"{feature}: available on {', '.join(plans)}")
            content = "Features Matrix: " + ". ".join(feature_parts)
            docs.append(Document(
                content=content,
                source_file="kb.yaml",
                citation="[kb.yaml#features_matrix]",
                metadata={"section": "features_matrix", "type": "features"}
            ))
        
        return docs
    
    def load_transcripts_json(self, filepath: str = "data/transcripts.json") -> List[Document]:
        """Load transcripts.json and create documents with transcript citations."""
        docs = []
        with open(filepath, 'r', encoding='utf-8') as f:
            transcripts = json.load(f)
        
        for transcript in transcripts:
            content = f"Customer Support Interaction ({transcript.get('channel', 'unknown')} channel, {transcript.get('region', 'unknown')} region): {transcript['text']}"
            citation = f"[transcripts.json#{transcript['id']}]"
            
            docs.append(Document(
                content=content,
                source_file="transcripts.json",
                citation=citation,
                metadata={"transcript_id": transcript['id'], "channel": transcript.get('channel'), "region": transcript.get('region')}
            ))
        
        return docs
    
    def load_faq_jsonl(self, filepath: str = "data/faq.jsonl") -> List[Document]:
        """Load faq.jsonl and create documents."""
        docs = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f, start=1):
                faq = json.loads(line.strip())
                content = f"Q: {faq['q']} A: {faq['a']}"
                citation = f"[faq.jsonl#entry={idx}]"
                
                docs.append(Document(
                    content=content,
                    source_file="faq.jsonl",
                    citation=citation,
                    metadata={"faq_id": idx, "question": faq['q']}
                ))
        
        return docs
    
    def load_all(self) -> List[Document]:
        """Load all data sources and return combined documents."""
        self.documents = []
        self.documents.extend(self.load_plans_csv())
        self.documents.extend(self.load_kb_yaml())
        self.documents.extend(self.load_transcripts_json())
        self.documents.extend(self.load_faq_jsonl())
        return self.documents
