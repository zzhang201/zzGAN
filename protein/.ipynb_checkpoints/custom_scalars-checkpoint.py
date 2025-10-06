import os
import json
from collections import defaultdict
from tensorboard.backend.event_processing import event_accumulator

def generate_dynamic_layout(logdir, run_name="run_001"):
    ea = event_accumulator.EventAccumulator(os.path.join(logdir, run_name))
    ea.Reload()
    tags = ea.Tags().get("scalars", [])

    grouped = defaultdict(list)
    for tag in tags:
        if "Anarci" in tag:
            grouped["Anarci Results"].append(tag)
        elif "Loss" in tag:
            grouped["Losses"].append(tag)
        elif "Gradient" in tag:
            grouped["Gradients"].append(tag)
        elif "JS_Divergence" in tag or "Diversity" in tag:
            grouped["Diversity Metrics"].append(tag)
        elif "Length" in tag:
            grouped["Lengths"].append(tag)
        elif "Attention" in tag:
            grouped["Attention Scores"].append(tag)
        elif "CosineSimilarity" in tag:
            grouped["T-SNE Analysis"].append(tag)
        else:
            grouped["Other"].append(tag)

    scalars_config = {
        "version": 0,
        "categories": []
    }

    for group_name, tag_list in grouped.items():
        scalars_config["categories"].append({
            "title": group_name,
            "chart": [{
                "type": "multiline",
                "title": group_name,
                "tag": tag_list
            }]
        })

    # Write to plugins/custom_scalar
    layout_path = os.path.join(logdir, "plugins", "custom_scalar", "custom_scalars_layout.json")
    os.makedirs(os.path.dirname(layout_path), exist_ok=True)

    with open(layout_path, "w") as f:
        json.dump(scalars_config, f, indent=2)

    print(f"Custom scalar layout written to {layout_path}")
