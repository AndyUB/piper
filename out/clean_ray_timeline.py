import argparse
import json

# Usage: python clean_ray_timeline.py <timeline_name>.json
# Output: <timeline_name>.cleaned.json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean Ray timeline JSON file")
    parser.add_argument("file", type=str, help="Path to the Ray timeline JSON file")
    args = parser.parse_args()

    file: str = args.file
    if not file.endswith(".json"):
        raise ValueError("Input file must be a JSON file")
    with open(file, "r") as f:
        events: list[dict] = json.load(f)

    # keep only entries for "forward", "backward", and "update" events
    # {
    #     "cat": "...",
    #     "name": "forward/backward/update",
    #     "pid": "10.158.48.71",
    #     "tid": "worker:efdca41b85ecacb4f64b5a8b5d98760091baa6e27f6a4877c7052e0a",
    #     "ts": 1769627046285342.0,
    #     "dur": 79146.695,
    #     "ph": "X",
    #     "cname": "generic_work",
    #     "args": { ... }
    # },

    filtered_events = []
    for event in events:
        if event.get("name") in ["forward", "backward", "update"]:
            filtered_events.append(event)

    cleaned_file = f"{file[:-5]}.cleaned.json"
    with open(cleaned_file, "w") as f:
        json.dump(filtered_events, f, indent=4)
