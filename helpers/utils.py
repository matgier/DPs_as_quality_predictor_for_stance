
import json

def open_json_file(path_to_file):
    try:
        # Read JSON file
        with open(path_to_file, "r") as file:
            texts = json.load(file)  # Ensure JSON is correctly formatted
        return texts
    except FileNotFoundError:
        print(f"Error: File not found at {path_to_file}")
    except json.JSONDecodeError:
        print("Error: Invalid JSON format.")

