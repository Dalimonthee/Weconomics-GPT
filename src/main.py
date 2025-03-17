import os
import argparse
import logging
from gemini_implementation import GeminiBookAgent, GeminiManagerAgent
import time
import json
from typing import List, Dict, Any
import sys
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("book_agent_system.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    """Load configuration from a JSON file."""
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from {config_path}")
            return config
        else:
            logger.warning(f"Config file {config_path} not found. Using defaults.")
            return {}
    except Exception as e:
        logger.error(f"Error loading config: {str(e)}")
        return {}

def save_history(question: str, answer: str, history_file: str = "question_history.jsonl"):
    """Save question and answer to history file."""
    entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "question": question,
        "answer": answer
    }
    try:
        with open(history_file, 'a') as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        logger.error(f"Error saving to history: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Book Agent System")
    parser.add_argument("--books_dir", default="books", help="Directory containing book files")
    parser.add_argument("--config", default="config.json", help="Path to configuration file")
    parser.add_argument("--history", default="question_history.jsonl", help="Path to question history file")
    parser.add_argument("--model", default="gemini-2.0-pro-exp", help="Model to use for agents")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    model_name = config.get("model_name", args.model)
    books_dir = config.get("books_dir", args.books_dir)
    history_file = config.get("history_file", args.history)
    
    logger.info(f"Starting Book Agent System with model {model_name}")
    
    try:
        # Check if books directory exists
        if not os.path.exists(books_dir):
            logger.error(f"Books directory {books_dir} does not exist")
            print(f"Error: Books directory {books_dir} not found. Please create it and add book files.")
            return
        
        # Get book paths
        book_paths = [os.path.join(books_dir, f) for f in os.listdir(books_dir) if f.endswith(".txt")]
        
        if not book_paths:
            logger.error(f"No book files found in {books_dir}")
            print(f"Error: No book files found in {books_dir}. Add some .txt files.")
            return
        
        # Create book agents
        start_time = time.time()
        print("Initializing book agents...")
        book_agents = [GeminiBookAgent(path, model_name) for path in book_paths]
        logger.info(f"Created {len(book_agents)} book agents in {time.time() - start_time:.2f}s")
        
        # Create manager agent
        print("Initializing manager agent...")
        manager = GeminiManagerAgent(book_agents, model_name)
        
        # Interactive loop
        print("\n===== Book Agent System =====")
        print("Type 'exit' to quit, 'help' for commands")
        print(f"Available books: {', '.join([agent.book_title for agent in book_agents])}")
        
        while True:
            try:
                query = input("\nYour question: ").strip()
                
                if not query:
                    continue
                
                if query.lower() == "exit":
                    print("Goodbye!")
                    break
                    
                if query.lower() == "help":
                    print("\nCommands:")
                    print("  exit - Exit the program")
                    print("  help - Show this help message")
                    print("  books - List available books")
                    print("  @BookTitle: question - Ask a specific book")
                    print("  @all: question - Ask all books and get a synthesized answer")
                    continue
                    
                if query.lower() == "books":
                    print("\nAvailable books:")
                    for i, agent in enumerate(book_agents, 1):
                        print(f"  {i}. {agent.book_title}")
                    continue
                
                # Handle directed questions
                specific_book = None
                require_all_books = False
                
                if query.startswith("@"):
                    parts = query.split(":", 1)
                    if len(parts) == 2:
                        target = parts[0][1:].strip()
                        question = parts[1].strip()
                        
                        if target.lower() == "all":
                            require_all_books = True
                            query = question
                        else:
                            # Find the closest matching book
                            for book_title in manager.book_agents.keys():
                                if target.lower() in book_title.lower():
                                    specific_book = book_title
                                    query = question
                                    break
                
                print("Processing your question...")
                start_time = time.time()
                
                answer = manager.route_question(
                    query, 
                    specific_book=specific_book,
                    require_all_books=require_all_books
                )
                
                print(f"\nAnswer (in {time.time() - start_time:.2f}s):")
                print(answer)
                
                # Save to history
                save_history(query, answer, history_file)
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                logger.error(f"Error: {str(e)}")
                print(f"Error: {str(e)}")
                traceback.print_exc()
    
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        print(f"Fatal error: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main() 