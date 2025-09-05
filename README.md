# Telegram Chat Search

This project is a simple command-line tool to search through messages from a Telegram group chat. It uses a TF-IDF based search to find relevant messages based on a query.

## Setup

1.  **Place your data**: Put your exported Telegram chat data (in JSON format) into the `statics/` directory and name it `result.json`. A sample file is provided.

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Command-line interface

To search for a message, run the `process.py` script with your query as a command-line argument.

```bash
python process.py "your search query"
```

### Web interface

To use the web interface, run the `app.py` script:

```bash
python app.py
```

Then, open your web browser and go to `http://127.0.0.1:5000`.

### Example

```bash
python process.py "استاد رضایی"
```

The first time you run the script, it will build a search index and cache it in the `.cache/` directory. Subsequent runs will be much faster as they will use the cached index.
