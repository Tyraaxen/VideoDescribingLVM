import json
import numpy as np

# Load the prediction and ground truth JSON files
with open("/home/taxen/VideoDescribingLVM/answers/openai_answers_es.json", "r") as f:
    predictions = json.load(f)

with open("/home/taxen/VideoDescribingLVM/qvhighlight/filtered_dataset.json", "r") as f:
    ground_truths = json.load(f)

# Convert ground truth list to a dictionary for easy lookup
gt_dict = {gt["vid"]: gt for gt in ground_truths}

# Define metric functions
def calculate_iou(pred, gt):
    """Calculate Intersection over Union (IoU)"""
    intersection = max(0, min(pred[1], gt[1]) - max(pred[0], gt[0]))
    union = (gt[1] - gt[0]) + (pred[1] - pred[0]) - intersection
    return intersection / union if union > 0 else 0

def temporal_coverage(pred, gt):
    """Calculate how much of the ground truth window was covered"""
    intersection = max(0, min(pred[1], gt[1]) - max(pred[0], gt[0]))
    return intersection / (gt[1] - gt[0]) if (gt[1] - gt[0]) > 0 else 0

def temporal_errors(pred, gt):
    """Calculate start and end time errors"""
    start_error = abs(pred[0] - gt[0])
    end_error = abs(pred[1] - gt[1])
    return start_error, end_error

# Store results
ious = []
coverages = []
start_errors = []
end_errors = []

# Compute metrics for each prediction
for pred in predictions:
    vid = pred["vid"]
    if vid in gt_dict:
        gt_entry = gt_dict[vid]
        gt_windows = gt_entry["relevant_windows"]

        # Compute IoU, coverage, and errors for the best-matching ground truth window
        best_iou = 0
        best_coverage = 0
        best_start_error = float("inf")
        best_end_error = float("inf")

        for gt_window in gt_windows:
            iou = calculate_iou(pred["answer"], gt_window)
            coverage = temporal_coverage(pred["answer"], gt_window)
            start_err, end_err = temporal_errors(pred["answer"], gt_window)

            if iou > best_iou:
                best_iou = iou
                best_coverage = coverage
                best_start_error = start_err
                best_end_error = end_err

        # Store best values
        ious.append(best_iou)
        coverages.append(best_coverage)
        start_errors.append(start_err)
        print("errors:" ,start_err, end_err)
        end_errors.append(end_err)

# Compute and display averages
average_iou = np.mean(ious)
average_coverage = np.mean(coverages)
average_start_error = np.mean(start_errors)
average_end_error = np.mean(end_errors)

# Print results
print("=== Evaluation Results ===")
print(f"Average IoU: {average_iou:.2f}")
print(f"Average Coverage: {average_coverage:.2f}")
print(f"Average Start Error: {average_start_error:.2f} sec")
print(f"Average End Error: {average_end_error:.2f} sec")
