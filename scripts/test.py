#!/usr/bin/env python3
import sys
import os
from datetime import datetime

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, '..')
sys.path.insert(0, os.path.abspath(project_root))

import torch
import glob
import numpy as np
from collections import defaultdict
from backend.app.model_fun.test_model import deviceLoader, modelLoader
from torch.utils.data import DataLoader
from backend.app.model_fun.train_model import ChunkDataset
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

CLASS_NAMES = ['O. exaltata', 'O. garganica', 'O. incubacea', 'O. majellensis', 'O. sphegodes', 'O. sphegodes_Palena']
CLASS_SIZE = len(CLASS_NAMES)

# Find all chunk files in the directory
base_dir = '../datasets/test/cropped_original'
abs_base_dir = os.path.abspath(base_dir)

print(f"Looking for all chunk files in: {abs_base_dir}")
all_chunks = glob.glob(os.path.join(abs_base_dir, "*.pt"))
print(f"Found {len(all_chunks)} chunk files:")
for chunk in all_chunks:
    print(f"  - {os.path.basename(chunk)}")

# Use the first chunk's base path
if all_chunks:
    first_chunk = all_chunks[0]
    base_path = first_chunk.replace('_chunk_0.pt', '').replace(abs_base_dir, base_dir)
    # Extract just the filename for reporting
    pt_filename = os.path.basename(first_chunk)
    print(f"\nUsing base path: {base_path}")
    print(f"PT file: {pt_filename}")
else:
    raise FileNotFoundError("No chunk files found!")

MODEL_PATH = '../backend/app/models/detection_models/global/model.pt'

if __name__ == '__main__':
    device = deviceLoader()
    model = modelLoader(MODEL_PATH, CLASS_SIZE, device)

    print("\nCaricamento dataset test...")
    testDataset = ChunkDataset(base_path, chunk_size=3000)
    print(f"Test dataset: {len(testDataset)} immagini")

    testDataLoader = DataLoader(
        testDataset, 
        batch_size=10, 
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    # Evaluation with F-Score and confusion matrix
    print("\nRunning evaluation...")
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_correct = 0
    all_total = 0
    
    # Per-class statistics
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    
    with torch.no_grad():
        for batch_idx, (images, labels, filenames) in enumerate(testDataLoader):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            # Store for metrics
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Accuracy
            all_total += labels.size(0)
            batch_correct = (predicted == labels).sum().item()
            all_correct += batch_correct
            
            # Per-class accuracy
            for i in range(len(labels)):
                label = labels[i].item()
                pred = predicted[i].item()
                class_total[label] += 1
                if label == pred:
                    class_correct[label] += 1
            
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}: {all_correct}/{all_total} correct ({100*all_correct/all_total:.1f}%)")
    
    # Convert to numpy arrays
    y_true = np.array(all_labels)
    y_pred = np.array(all_predictions)
    
    # Calculate metrics
    # Overall Accuracy
    accuracy = all_correct / all_total
    
    # F-Score
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
    
    # Build report string
    report_lines = []
    report_lines.append("="*70)
    report_lines.append(" TEST REPORT")
    report_lines.append("="*70)
    report_lines.append(f" Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f" Dataset file: {pt_filename}")
    report_lines.append(f" Total images: {len(testDataset)}")
    report_lines.append("="*70)
    report_lines.append("")
    report_lines.append(" RISULTATI")
    report_lines.append("="*70)
    report_lines.append("")
    report_lines.append(f"Accuracy: {accuracy:.4f} ({all_correct}/{all_total})")
    report_lines.append("")
    report_lines.append("F-Score:")
    report_lines.append(f"  Macro F1:     {f1_macro:.4f}")
    report_lines.append(f"  Weighted F1:  {f1_weighted:.4f}")
    report_lines.append("")
    report_lines.append("  Per-class F1:")
    for i, class_name in enumerate(CLASS_NAMES):
        if i < len(f1_per_class):
            report_lines.append(f"    {class_name:<20}: {f1_per_class[i]:.4f}")
    
    report_lines.append("")
    report_lines.append("Per-class Accuracy:")
    for i, class_name in enumerate(CLASS_NAMES):
        if class_total[i] > 0:
            acc = class_correct[i] / class_total[i]
            report_lines.append(f"  {class_name:<20}: {acc:.4f} ({class_correct[i]}/{class_total[i]})")
        else:
            report_lines.append(f"  {class_name:<20}: N/A (no samples)")
    
    report_lines.append("")
    report_lines.append("Confusion Matrix:")
    report_lines.append("")
    
    # Header row
    header = f"{'True\\Pred':<15}"
    for name in CLASS_NAMES:
        header += f"{name[:8]:>8}"
    report_lines.append(header)
    report_lines.append("-" * 70)
    
    # Matrix rows
    for i, true_name in enumerate(CLASS_NAMES):
        row = f"{true_name:<15}"
        for j in range(len(CLASS_NAMES)):
            row += f"{cm[i,j]:>8}"
        report_lines.append(row)
    
    report_lines.append("")
    report_lines.append("="*70)
    report_lines.append(" CLASSIFICATION REPORT")
    report_lines.append("="*70)
    report_lines.append(classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0))
    
    # Join all lines
    full_report = "\n".join(report_lines)
    
    # Print to console
    print("\n" + full_report)
    
    # Save to file with date
    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report_filename = f"report_{date_str}.txt"
    
    with open(report_filename, 'w') as f:
        f.write(full_report)
    
    print(f"\nReport saved to: {report_filename}")
    
    # Plot and save confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=[c[:10] for c in CLASS_NAMES],
                yticklabels=[c[:10] for c in CLASS_NAMES])
    plt.title(f'Confusion Matrix\nDataset: {pt_filename}', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.tight_layout()
    
    cm_filename = f"confusion_matrix_{date_str}.png"
    plt.savefig(cm_filename, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to: {cm_filename}")
