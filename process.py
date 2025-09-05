import argparse
from src.data_processing import load_messages, group_messages_into_threads
from src.search_engine import SearchEngine

def main():
    parser = argparse.ArgumentParser(description="Search through chat messages.")
    parser.add_argument("query", type=str, help="The search query.")
    args = parser.parse_args()

    # 1. Initialize the search engine
    search_engine = SearchEngine()

    # 2. Check if an index exists. If not, build it.
    if not search_engine.has_index():
        print("No cached index found. Building a new one...")
        messages = load_messages()
        threads = group_messages_into_threads(messages)
        search_engine.build_index(threads)
        print("New index built and cached.")
    else:
        print("Loaded search index from cache.")

    # 3. Perform the search
    results = search_engine.search(args.query)

    # 4. Display results
    print(f"\nQuery: {args.query}")
    if results:
        for result in results:
            print(f"Answer: Based on message {result['message_id']} → \"{result['text']}\"")
    else:
        print("No results found.")

if __name__ == "__main__":
    main()
