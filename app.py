from flask import Flask, render_template, request, jsonify
from src.data_processing import load_messages, group_messages_into_threads
from src.search_engine import SearchEngine

app = Flask(__name__, template_folder='templates')

# Initialize the search engine
search_engine = SearchEngine()

# Check if an index exists. If not, build it.
if not search_engine.has_index():
    print("No cached index found. Building a new one...")
    messages = load_messages()
    threads = group_messages_into_threads(messages)
    search_engine.build_index(threads)
    print("New index built and cached.")
else:
    print("Loaded search index from cache.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    data = request.get_json()
    query = data.get('query')
    if not query:
        return jsonify({"error": "Query is required"}), 400

    results = search_engine.search(query)
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
