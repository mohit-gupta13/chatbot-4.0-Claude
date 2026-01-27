import urllib.request
import json
import time
import random

def test_logging():
    url = "http://127.0.0.1:8000/chat"
    
    # Check if server is reachable
    try:
        urllib.request.urlopen("http://127.0.0.1:8000/", timeout=1)
    except Exception as e:
        print(f"Error: Server not reachable at http://127.0.0.1:8000. {e}")
        return

    # Generate a unique unique query
    unique_id = int(time.time())
    unique_query = f"unique_query_checking_logging_{unique_id}"
    print(f"Sending query: {unique_query}")
    
    data = json.dumps({"message": unique_query}).encode('utf-8')
    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    
    try:
        with urllib.request.urlopen(req) as response:
            resp_data = json.loads(response.read().decode('utf-8'))
            print(f"Server Response: {resp_data}")
    except Exception as e:
        print(f"Request failed: {e}")
        return

    # Wait a moment for file write
    time.sleep(1)

    # Check the log file
    try:
        import os
        abs_path = os.path.abspath("unanswered_queries.json")
        print(f"Reading log file from: {abs_path}")
        with open("unanswered_queries.json", "r") as f:
            logs = json.load(f)
            # Check if our query is in the logs
            found = False
            for entry in logs:
                if entry.get("question") == unique_query:
                    print("\nSUCCESS: Query found in log file!")
                    print(entry)
                    found = True
                    break
            
            if not found:
                print("\nFAILURE: Query NOT found in log file.")
                print("Last 3 entries in log:")
                print(json.dumps(logs[-3:], indent=2))
                
    except Exception as e:
        print(f"Error reading log file: {e}")

if __name__ == "__main__":
    test_logging()
