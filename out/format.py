import json
import sys

if __name__ == "__main__":
    if len(sys.argv) > 1:
        clean = sys.argv[1] == "clean"
    else:
        clean = False

    # file = "LLAMA_3B-pp2-dp1-interleaved-1f1b-v3.dashboard"
    # file = "LLAMA_3B-pp2-dp1-1f1b-min_v3"
    file = "LLAMA_3B-pp2-dp1-interleaved-1f1b-sync_v2"
    with open(f"{file}.json", "r") as f:
        data = json.load(f)

    if clean:
        # data is a list of dicts
        # keep only entries like:
        #  {
        #     "cat": "task::fwd_s2_mb3",
        #     "name": "forward",
        #     "pid": "10.158.48.71",
        #     "tid": "worker:efdca41b85ecacb4f64b5a8b5d98760091baa6e27f6a4877c7052e0a",
        #     "ts": 1769627046285342.0,
        #     "dur": 79146.695,
        #     "ph": "X",
        #     "cname": "generic_work",
        #     "args": {
        #         "name": "forward",
        #         "task_id": "44676a208cefd2e670eb10b0e33348655a1a293501000000"
        #     }
        # },
        cleaned_events = []
        for event in data:
            if "name" in event and (
                event["name"] == "forward" or event["name"] == "backward"
            ):
                cleaned_events.append(event)
        with open(f"{file}.cleaned.json", "w") as f:
            json.dump(cleaned_events, f, indent=4)
    else:
        # save pretty-printed json to file
        with open(f"{file}.pretty.json", "w") as f:
            json.dump(data, f, indent=4)
