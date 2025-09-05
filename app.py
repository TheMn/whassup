import shutil
from flask import Flask, render_template, request, jsonify
from src.data_processing import load_data, group_messages_into_threads
from src.search_engine import SearchEngine, CACHE_DIR

app = Flask(__name__, template_folder='templates')

search_engine = SearchEngine()

def build_search_index():
    """Loads data and builds the search index."""
    print("Building search index...")
    messages, _ = load_data()
    threads = group_messages_into_threads(messages)
    search_engine.build_index(threads)
    print("New index built and cached.")

# Initial index build if not exists
if not search_engine.has_index():
    build_search_index()
else:
    print("Loaded search index from cache.")

# Load chat_id for generating links
_, chat_id = load_data()

@app.route('/')
def index():
    return render_template('index.html', chat_id=chat_id)

@app.route('/reset-cache', methods=['POST'])
def reset_cache():
    """Clears the cache and rebuilds the index."""
    print("Resetting cache...")
    try:
        shutil.rmtree(CACHE_DIR)
        print("Cache directory removed.")
        build_search_index()
        return jsonify({"message": "Cache reset and index rebuilt successfully."})
    except FileNotFoundError:
        print("Cache directory not found, building index anyway.")
        build_search_index()
        return jsonify({"message": "Cache was already empty. Index rebuilt."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/search', methods=['POST'])
def search():
    data = request.get_json()
    query = data.get('query')
    if not query:
        return jsonify({"error": "Query is required"}), 400

    results = search_engine.search(query)
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
