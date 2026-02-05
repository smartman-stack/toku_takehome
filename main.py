"""
Main entry point for TokuTel Assistant.
Supports interactive mode, evaluation, and various configuration options.
"""
import sys
import os
from assistant import TokuTelAssistant, create_assistant
from evaluate import evaluate
from config import get_config


def print_banner():
    """Print welcome banner."""
    print("""
╔═══════════════════════════════════════════════════════════════╗
║                   TokuTel RAG Assistant                       ║
║         Customer Support powered by AI                        ║
╚═══════════════════════════════════════════════════════════════╝
""")


def print_help():
    """Print help information."""
    print("""
Commands:
  help     - Show this help message
  eval     - Run evaluation on eval_prompts.txt
  explain  - Toggle explanation mode (shows retrieval details)
  status   - Show assistant status
  clear    - Clear screen
  exit     - Exit the assistant

Options:
  --offline     Run without LLM (template-based generation)
  --no-rerank   Disable cross-encoder reranking
  --eval        Run evaluation mode directly

Environment Variables:
  OPENAI_API_KEY    - OpenAI API key for LLM-based generation
  ANTHROPIC_API_KEY - Anthropic API key (alternative)
  LLM_PROVIDER      - "openai", "anthropic", or "offline"
""")


def interactive_mode(assistant: TokuTelAssistant):
    """Run assistant in interactive mode."""
    print_banner()
    print("Type 'help' for available commands\n")
    
    assistant.initialize()
    
    explain_mode = False
    
    while True:
        try:
            query = input("\n🎯 Customer Question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye!")
            break
        
        if not query:
            continue
        
        # Handle commands
        if query.lower() == 'exit' or query.lower() == 'quit':
            print("Goodbye!")
            break
        
        if query.lower() == 'help':
            print_help()
            continue
        
        if query.lower() == 'eval':
            print("\nRunning evaluation...\n")
            evaluate()
            continue
        
        if query.lower() == 'explain':
            explain_mode = not explain_mode
            print(f"Explanation mode: {'ON' if explain_mode else 'OFF'}")
            continue
        
        if query.lower() == 'status':
            status = assistant.get_status()
            print("\n📊 Assistant Status:")
            print(f"   Initialized: {status['initialized']}")
            print(f"   LLM Available: {status['config']['llm_available']}")
            print(f"   Reranker: {'Enabled' if status['config']['reranker_enabled'] else 'Disabled'}")
            print(f"   Documents: {status['documents_indexed']}")
            continue
        
        if query.lower() == 'clear':
            os.system('cls' if os.name == 'nt' else 'clear')
            print_banner()
            continue
        
        # Generate answer
        print("\n" + "─" * 60)
        
        if explain_mode:
            result = assistant.explain_answer(query)
            
            print("\n📄 Retrieved Documents:")
            for i, doc in enumerate(result['breakdown']['retrieved_documents'][:5], 1):
                print(f"   {i}. [{doc['type']}] {doc['citation']} (score: {doc['score']})")
            
            print(f"\n🔧 Method: {result['breakdown']['generation_method']}")
            print(f"📊 Confidence: {result['breakdown']['confidence']:.2%}")
        else:
            result = assistant.generate_answer(query)
        
        print("\n💬 Answer:")
        print(result["answer"])
        
        print("\n📚 Citations:", ", ".join(result.get("citations", [])))
        
        if result.get("escalation", {}).get("needed"):
            print(f"\n⚠️  {result['escalation']['message']}")
        
        print("\n" + "─" * 60)


def main():
    """Main entry point."""
    # Parse arguments
    args = sys.argv[1:]
    
    use_llm = '--offline' not in args
    use_reranker = '--no-rerank' not in args
    run_eval = '--eval' in args or 'eval' in args
    
    # Create assistant with configuration
    assistant = create_assistant(use_llm=use_llm, use_reranker=use_reranker)
    
    if run_eval:
        # Run evaluation mode
        assistant.initialize()
        evaluate()
    else:
        # Run interactive mode
        interactive_mode(assistant)


if __name__ == "__main__":
    main()
