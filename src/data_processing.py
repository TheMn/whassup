import json
from collections import defaultdict
import os

# Get the absolute path to the project root
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_FILEPATH = os.path.join(ROOT_DIR, "statics", "result.json")

def load_data(filepath=DEFAULT_FILEPATH):
    """Loads messages and chat ID from the JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    chat_id = data.get("id")
    messages = []
    for msg in data["messages"]:
        if msg["type"] == "message" and "text" in msg:
            messages.append({
                "id": msg["id"],
                "text": msg["text"] if isinstance(msg["text"], str) else "".join([t if isinstance(t, str) else "" for t in msg["text"]]),
                "reply_to": msg.get("reply_to_message_id"),
                "from": msg.get("from", "Unknown Sender"),
                "date": msg.get("date"),
            })
    return messages, chat_id

def group_messages_into_threads(messages):
    """Groups messages into threads based on reply_to."""
    threads = defaultdict(list)
    message_map = {msg['id']: msg for msg in messages}

    for msg in messages:
        if msg["reply_to"] and msg["reply_to"] in message_map:
            root_id = msg["reply_to"]
            # Traverse up to find the ultimate root of the thread
            while message_map[root_id]["reply_to"] and message_map[root_id]["reply_to"] in message_map:
                root_id = message_map[root_id]["reply_to"]
            threads[root_id].append(msg)
        else:
            # This message is a root of a new thread
            threads[msg["id"]].append(msg)

    # Add the root message to each thread
    for root_id, thread_messages in threads.items():
        if root_id in message_map:
            root_message = message_map[root_id]
            if root_message not in thread_messages:
                thread_messages.insert(0, root_message)

    return threads
