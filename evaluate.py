import json

# Load the ground truth answers
with open("/home/taxen/Desktop/MasterThesis/VideoDescribingLVM/Questions/test.json", "r") as f:
    ground_truth = json.load(f)

# Load the model's predictions
with open("/home/taxen/Desktop/MasterThesis/VideoDescribingLVM/Questions/test_answers.json", "r") as f:
    model_answers = json.load(f)

# Convert ground truth to a dictionary for quick lookup
ground_truth_dict = {(entry["video_id"]): entry["answer"] for entry in ground_truth}
# Compare model answers with ground truth
correct_count = 0
total_answers = 0

for model_entry in model_answers:
    key = (model_entry["video_id"])
    if key in ground_truth_dict:  # Ensure matching question-video pair exists
        total_answers += 1
        if model_entry["answer"] == ground_truth_dict[key]:
            correct_count += 1

# Calculate accuracy
accuracy = (correct_count / total_answers) * 100 if total_answers > 0 else 0

# Print results
print(f"Correct Answers: {correct_count}/{total_answers}")
print(f"Accuracy: {accuracy:.2f}%")
