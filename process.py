import argparse
from src.data_processing import load_messages, group_messages_into_threads
from src.search_engine import SearchEngine

def main():
    parser = argparse.ArgumentParser(description="Search through chat messages.")
    parser.add_argument("query", type=str, help="The search query.")
    args = parser.parse_args()

    # 1. Load and process data
    messages = load_messages()
    threads = group_messages_into_threads(messages)

    # 2. Build the search index
    search_engine = SearchEngine()
    search_engine.build_index(threads)

    # 3. Perform the search
    results = search_engine.search(args.query)

    # 4. Display results
    print(f"Query: {args.query}")
    if results:
        for result in results:
            # The search result now returns a dictionary with message_id and text
            print(f"Answer: Based on message {result['message_id']} → \"{result['text']}\"")
    else:
        print("No results found.")

if __name__ == "__main__":
    main()
