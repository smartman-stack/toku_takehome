"""
Evaluation script for running the assistant on eval_prompts.txt
"""
from assistant import TokuTelAssistant
import json
from datetime import datetime
from typing import List


def load_eval_prompts(filepath: str = "data/eval_prompts.txt") -> List[str]:
    """Load evaluation prompts."""
    with open(filepath, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts


def evaluate():
    """Run evaluation and save outputs."""
    print("Initializing TokuTel Assistant...")
    assistant = TokuTelAssistant()
    assistant.initialize()
    
    print("\nLoading evaluation prompts...")
    prompts = load_eval_prompts()
    
    print(f"\nEvaluating on {len(prompts)} prompts...\n")
    
    results = []
    
    for idx, prompt in enumerate(prompts, 1):
        print(f"Processing prompt {idx}/{len(prompts)}: {prompt[:60]}...")
        
        response = assistant.generate_answer(prompt, top_k=5)
        
        result = {
            "prompt_id": idx,
            "query": prompt,
            "answer": response["answer"],
            "citations": response["citations"],
            "escalation": response["escalation"],
            "sla_guidance": response["sla_guidance"],
            "retrieved_docs": response["retrieved_docs"]
        }
        
        results.append(result)
        
        # Print summary
        print(f"  Answer length: {len(response['answer'])} chars")
        print(f"  Citations: {len(response['citations'])}")
        print(f"  Escalation needed: {response['escalation']['needed']}")
        print()
    
    # Save results
    output_file = "evaluation_outputs.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "total_prompts": len(prompts),
            "results": results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\nEvaluation complete! Results saved to {output_file}")
    
    # Also create a human-readable output
    output_file_txt = "evaluation_outputs.txt"
    with open(output_file_txt, 'w', encoding='utf-8') as f:
        f.write("TokuTel Assistant Evaluation Results\n")
        f.write("=" * 60 + "\n\n")
        
        for result in results:
            f.write(f"Prompt {result['prompt_id']}: {result['query']}\n")
            f.write("-" * 60 + "\n")
            f.write(f"Answer:\n{result['answer']}\n\n")
            f.write(f"Citations: {', '.join(result['citations'])}\n")
            if result['escalation']['needed']:
                f.write(f"Escalation: {result['escalation']['message']}\n")
            f.write(f"SLA Guidance: {result['sla_guidance']}\n")
            f.write("\n" + "=" * 60 + "\n\n")
    
    print(f"Human-readable results saved to {output_file_txt}")
    
    return results


if __name__ == "__main__":
    evaluate()
