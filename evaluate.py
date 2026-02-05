"""
Evaluation script for running the assistant on eval_prompts.txt.
Produces structured JSON and human-readable outputs.
"""
import json
from datetime import datetime
from typing import List, Dict, Any
from assistant import TokuTelAssistant
from config import get_config


def load_eval_prompts(filepath: str = "data/eval_prompts.txt") -> List[str]:
    """Load evaluation prompts from file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts


def evaluate(assistant: TokuTelAssistant = None, 
             output_json: str = "evaluation_outputs.json",
             output_txt: str = "evaluation_outputs.txt") -> List[Dict[str, Any]]:
    """
    Run evaluation on all prompts and save results.
    
    Args:
        assistant: TokuTelAssistant instance (creates new if None)
        output_json: Path for JSON output
        output_txt: Path for human-readable output
    
    Returns:
        List of evaluation results
    """
    # Initialize assistant if not provided
    if assistant is None:
        print("Initializing TokuTel Assistant for evaluation...")
        assistant = TokuTelAssistant()
        assistant.initialize()
    
    # Load prompts
    config = get_config()
    prompts = load_eval_prompts(config.eval_prompts_file)
    
    print(f"\n{'='*60}")
    print(f"Running evaluation on {len(prompts)} prompts")
    print(f"{'='*60}\n")
    
    results = []
    
    for idx, prompt in enumerate(prompts, 1):
        print(f"[{idx}/{len(prompts)}] Processing: {prompt[:50]}...")
        
        # Generate response
        response = assistant.generate_answer(prompt, top_k=5)
        
        result = {
            "prompt_id": idx,
            "query": prompt,
            "answer": response["answer"],
            "citations": response["citations"],
            "escalation": response["escalation"],
            "sla_guidance": response["sla_guidance"],
            "confidence": response.get("confidence", 0),
            "retrieved_docs": response["retrieved_docs"],
            "method": response.get("method", "unknown")
        }
        
        results.append(result)
        
        # Print progress
        print(f"         [OK] Answer: {len(response['answer'])} chars")
        print(f"         [OK] Citations: {len(response['citations'])}")
        print(f"         [OK] Confidence: {min(response.get('confidence', 0), 1.0):.2%}")
        print(f"         [OK] Method: {response.get('method', 'unknown')}")
        if response["escalation"]["needed"]:
            print(f"         [!] Escalation: {response['escalation']['level']}")
        print()
    
    # Save JSON output
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "total_prompts": len(prompts),
        "assistant_config": {
            "llm_available": assistant.generator._llm_available if assistant.generator else False,
            "reranker_enabled": assistant.config.model.use_reranker,
            "hybrid_search": True
        },
        "results": results
    }
    
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"[OK] JSON results saved to: {output_json}")
    
    # Save human-readable output
    with open(output_txt, 'w', encoding='utf-8') as f:
        f.write("TokuTel Assistant Evaluation Results\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        
        for result in results:
            f.write(f"Prompt {result['prompt_id']}: {result['query']}\n")
            f.write("-" * 60 + "\n")
            f.write(f"Answer:\n{result['answer']}\n\n")
            f.write(f"Citations: {', '.join(result['citations'])}\n")
            f.write(f"Confidence: {min(result['confidence'], 1.0):.2%}\n")
            f.write(f"Method: {result['method']}\n")
            if result['escalation']['needed']:
                f.write(f"Escalation: {result['escalation']['message']}\n")
            f.write(f"SLA Guidance: {result['sla_guidance']}\n")
            f.write("\n" + "=" * 60 + "\n\n")
    
    print(f"[OK] Human-readable results saved to: {output_txt}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total prompts:     {len(prompts)}")
    print(f"Avg confidence:    {sum(r['confidence'] for r in results)/len(results):.2%}")
    print(f"Avg citations:     {sum(len(r['citations']) for r in results)/len(results):.1f}")
    print(f"Escalations:       {sum(1 for r in results if r['escalation']['needed'])}")
    print(f"{'='*60}\n")
    
    return results


def compare_results(old_file: str, new_file: str):
    """Compare two evaluation result files."""
    with open(old_file, 'r') as f:
        old_results = json.load(f)
    with open(new_file, 'r') as f:
        new_results = json.load(f)
    
    print("\n📊 Comparison Report")
    print("=" * 60)
    
    for old_r, new_r in zip(old_results['results'], new_results['results']):
        print(f"\nPrompt {old_r['prompt_id']}: {old_r['query'][:40]}...")
        print(f"  Old citations: {len(old_r['citations'])} -> New: {len(new_r['citations'])}")
        print(f"  Old confidence: N/A -> New: {new_r.get('confidence', 0):.2%}")
        
        # Check answer length change
        old_len = len(old_r['answer'])
        new_len = len(new_r['answer'])
        change = ((new_len - old_len) / old_len * 100) if old_len > 0 else 0
        print(f"  Answer length: {old_len} -> {new_len} ({change:+.1f}%)")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "compare":
        if len(sys.argv) >= 4:
            compare_results(sys.argv[2], sys.argv[3])
        else:
            print("Usage: python evaluate.py compare <old_file.json> <new_file.json>")
    else:
        evaluate()
