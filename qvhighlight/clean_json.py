import json

# Input and output file paths
input_file = "qvhighlight/highlight_train_release.jsonl"  # Replace with your actual input file path
output_file = "qvhighlight/filtered_dataset.json"

# Read the dataset and filter it
filtered_data = []
with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line.strip())  # Parse each JSON object

        # Check if all saliency scores are >= 3
        if all(all(score >= 2 for score in clip) for clip in obj["saliency_scores"]):
            # Condition 2: Keep only objects with exactly **one** relevant_window
            if len(obj["relevant_windows"]) == 1:
                filtered_data.append(obj)  # Keep this object

# Save the filtered data as a valid JSON array
with open(output_file, "w", encoding="utf-8") as f_out:
    json.dump(filtered_data, f_out, indent=4)  # Save as a properly formatted JSON list

print(f"Filtered dataset saved to {output_file} with {len(filtered_data)} entries.")

