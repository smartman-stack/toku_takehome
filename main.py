"""
Main entry point for TokuTel Assistant.
Can be used for interactive queries or evaluation.
"""
import sys
from assistant import TokuTelAssistant
from evaluate import evaluate


def interactive_mode():
    """Run assistant in interactive mode."""
    print("TokuTel Assistant - Interactive Mode")
    print("Type 'exit' to quit, 'eval' to run evaluation\n")
    
    assistant = TokuTelAssistant()
    assistant.initialize()
    
    while True:
        query = input("\nCustomer Question: ").strip()
        
        if query.lower() == 'exit':
            break
        
        if query.lower() == 'eval':
            evaluate()
            continue
        
        if not query:
            continue
        
        response = assistant.generate_answer(query)
        
        print("\n" + "=" * 60)
        print("Answer:")
        print(response["answer"])
        print("\nCitations:", ", ".join(response["citations"]))
        if response["escalation"]["needed"]:
            print("\n⚠️", response["escalation"]["message"])
        print("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "eval":
        evaluate()
    else:
        interactive_mode()
