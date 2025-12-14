"""
Evaluation and visualization for QuickDraw Sketch Classification
Here there are metrics, confusion matrix, per-class analysis, predictions visualization
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix, classification_report
from typing import Dict, List, Optional, Any
from src.utils import get_timestamp


def evaluate_model(
    model: keras.Model,
    test_ds: tf.data.Dataset,
    class_names: List[str]
) -> Dict[str, float]:
    print("\n[INFO] Evaluating model on test set...")
    results = model.evaluate(test_ds, verbose=1, return_dict=True)
    
    print("TEST SET RESULTS")
    print(f"{'='*70}")
    for name, value in results.items():
        print(f"  {name:20s}: {value:.4f}")
    print(f"{'='*70}\n")
    
    return results


def get_predictions(
    model: keras.Model,
    dataset: tf.data.Dataset
) -> tuple:
    y_true_list = []
    y_pred_list = []
    y_prob_list = []
    
    for images, labels in dataset:
        probs = model.predict(images, verbose=0)
        preds = np.argmax(probs, axis=1)
        true = np.argmax(labels.numpy(), axis=1)
        
        y_true_list.extend(true)
        y_pred_list.extend(preds)
        y_prob_list.extend(probs)
    
    return (
        np.array(y_true_list),
        np.array(y_pred_list),
        np.array(y_prob_list)
    )


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
    normalize: bool = True
):
    plt.ioff()  
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  
        cm = cm.astype('float') / row_sums
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Proportion' if normalize else 'Count'}
    )
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Confusion matrix saved to {save_path}")
    
    plt.close()  


def plot_per_class_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None
):
    plt.ioff()  

    accuracies = []
    for i in range(len(class_names)):
        mask = y_true == i
        if mask.sum() > 0:
            acc = (y_pred[mask] == y_true[mask]).mean()
            accuracies.append(acc)
        else:
            accuracies.append(0)
    
    sorted_indices = np.argsort(accuracies)
    sorted_names = [class_names[i] for i in sorted_indices]
    sorted_accs = [accuracies[i] for i in sorted_indices]
    
    plt.figure(figsize=(12, 8))
    colors = plt.cm.RdYlGn(np.array(sorted_accs))
    bars = plt.barh(sorted_names, sorted_accs, color=colors)
    
    plt.xlabel('Accuracy', fontsize=12, fontweight='bold')
    plt.ylabel('Class', fontsize=12, fontweight='bold')
    plt.title('Per-Class Accuracy', fontsize=16, fontweight='bold', pad=20)
    plt.xlim(0, 1)
    plt.axvline(x=np.mean(accuracies), color='red', linestyle='--', 
                linewidth=2, label=f'Mean: {np.mean(accuracies):.3f}')
    plt.legend(fontsize=11)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Per-class accuracy plot saved to {save_path}")
    
    plt.close()  


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None
):
    unique_classes = np.unique(np.concatenate([y_true, y_pred]))
    labels = sorted(unique_classes)
    
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=[class_names[i] for i in labels],
        zero_division=0,
        digits=4
    )
    
    print("\nCLASSIFICATION REPORT")
    print("="*70)
    print(report)
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write("CLASSIFICATION REPORT\n")
            f.write("="*70 + "\n")
            f.write(report)
        print(f"[INFO] Classification report saved to {save_path}")


def visualize_predictions(
    model: keras.Model,
    test_ds: tf.data.Dataset,
    class_names: List[str],
    samples_per_class: int = 5,
    save_path: Optional[str] = None
):
    plt.ioff()  

    samples = {cls: [] for cls in class_names}
    
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        pred_idx = np.argmax(preds, axis=1)
        true_idx = np.argmax(labels.numpy(), axis=1)
        
        for i in range(len(images)):
            cls = class_names[true_idx[i]]
            if len(samples[cls]) < samples_per_class:
                samples[cls].append((
                    images[i].numpy(),
                    true_idx[i],
                    pred_idx[i],
                    preds[i][pred_idx[i]]
                ))
        
        if all(len(v) >= samples_per_class for v in samples.values()):
            break
    
    num_classes = len(class_names)
    fig = plt.figure(figsize=(15, 3 * num_classes))
    
    for row, cls in enumerate(class_names):
        items = samples[cls]
        for col in range(min(len(items), samples_per_class)):
            img, true_i, pred_i, conf = items[col]
            
            ax = plt.subplot(num_classes, samples_per_class, 
                           row * samples_per_class + col + 1)
            plt.imshow(img)
            plt.axis("off")
            
            pred_label = class_names[pred_i]
            true_label = class_names[true_i]
            
            color = 'green' if pred_i == true_i else 'red'
            title = f"P: {pred_label}\nT: {true_label}\n({conf:.2f})"
            plt.title(title, fontsize=9, color=color, fontweight='bold')
    
    plt.suptitle('Sample Predictions (Green=Correct, Red=Wrong)', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Predictions visualization saved to {save_path}")
    
    plt.close() 


def plot_top_errors(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    top_n: int = 10,
    save_path: Optional[str] = None
):
    plt.ioff() 

    errors = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j:
                count = ((y_true == i) & (y_pred == j)).sum()
                if count > 0:
                    errors.append((class_names[i], class_names[j], count))
    
    errors.sort(key=lambda x: x[2], reverse=True)
    top_errors = errors[:top_n]
    
    plt.figure(figsize=(12, 8))
    labels = [f"{true} → {pred}" for true, pred, _ in top_errors]
    counts = [count for _, _, count in top_errors]
    
    bars = plt.barh(labels, counts, color='coral')
    plt.xlabel('Number of Misclassifications', fontsize=12, fontweight='bold')
    plt.ylabel('True Class → Predicted Class', fontsize=12, fontweight='bold')
    plt.title(f'Top {top_n} Misclassification Pairs', 
             fontsize=16, fontweight='bold', pad=20)
    plt.gca().invert_yaxis()
    
    for bar in bars:
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height()/2, 
                f' {int(width)}', ha='left', va='center', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Top errors plot saved to {save_path}")
    
    plt.close()  


def comprehensive_evaluation(
    model: keras.Model,
    test_ds: tf.data.Dataset,
    class_names: List[str],
    config: Dict[str, Any]
):
    
    print("[INFO] Generating predictions...")
    y_true, y_pred, y_prob = get_predictions(model, test_ds)
    
    results = evaluate_model(model, test_ds, class_names)
    
    timestamp = get_timestamp()
    plots_dir = config['output']['plots_dir']
    logs_dir = config['output']['logs_dir']
    
    if config['evaluation']['confusion_matrix']:
        cm_path = os.path.join(plots_dir, f'confusion_matrix_{timestamp}.png')
        plot_confusion_matrix(y_true, y_pred, class_names, cm_path)
    
    if config['evaluation']['per_class_metrics']:
        acc_path = os.path.join(plots_dir, f'per_class_accuracy_{timestamp}.png')
        plot_per_class_accuracy(y_true, y_pred, class_names, acc_path)
        
        report_path = os.path.join(logs_dir, f'classification_report_{timestamp}.txt')
        print_classification_report(y_true, y_pred, class_names, report_path)
    
    if config['evaluation']['visualize_predictions']:
        viz_path = os.path.join(plots_dir, f'sample_predictions_{timestamp}.png')
        visualize_predictions(
            model, test_ds, class_names,
            config['evaluation']['samples_per_class_viz'],
            viz_path
        )
    
    errors_path = os.path.join(plots_dir, f'top_errors_{timestamp}.png')
    plot_top_errors(y_true, y_pred, class_names, top_n=10, save_path=errors_path)
    
    print("\n[INFO] Comprehensive evaluation completed!")
