"""
Util functions for QuickDraw Sketch Classification
Here tehre are seed setting, logging, plotting, and helper functions
"""

import os
import random
import yaml
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, Any, Optional


def set_seed(seed: int = 42):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    print(f"[INFO] Random seed set to {seed}")


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    print(f"[INFO] Configuration loaded from {config_path}")
    return config


def create_output_dirs(config: Dict[str, Any]):
    dirs = [
        config['output']['models_dir'],
        config['output']['logs_dir'],
        config['output']['plots_dir'],
        config['output']['history_dir']
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    
    print(f"[INFO] Output directories created/verified")


def get_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_history(history: Dict[str, list], filepath: str):
    with open(filepath, 'wb') as f:
        pickle.dump(history, f)
    print(f"[INFO] Training history saved to {filepath}")


def load_history(filepath: str) -> Dict[str, list]:
    with open(filepath, 'rb') as f:
        history = pickle.load(f)
    print(f"[INFO] Training history loaded from {filepath}")
    return history


def plot_training_history(history: Dict[str, list], save_path: Optional[str] = None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot accuracy
    axes[0].plot(history['accuracy'], label='Train Accuracy', linewidth=2)
    axes[0].plot(history['val_accuracy'], label='Val Accuracy', linewidth=2)
    axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Plot loss
    axes[1].plot(history['loss'], label='Train Loss', linewidth=2)
    axes[1].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Training history plot saved to {save_path}")
    
    plt.show()


def plot_learning_rate(history: Dict[str, list], save_path: Optional[str] = None):
    if 'lr' not in history:
        print("[WARNING] Learning rate not found in history")
        return
    
    plt.figure(figsize=(10, 5))
    plt.plot(history['lr'], linewidth=2, color='darkblue')
    plt.title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Learning Rate', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Learning rate plot saved to {save_path}")
    
    plt.show()


def count_images(root_dir: str) -> tuple:
    counts = {}
    total = 0
    
    for class_name in sorted(os.listdir(root_dir)):
        class_path = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        
        n_images = sum(1 for f in os.listdir(class_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg')))
        counts[class_name] = n_images
        total += n_images
    
    return counts, total


def print_dataset_info(class_counts: Dict[str, int], split_name: str = "Dataset"):
    total = sum(class_counts.values())
    num_classes = len(class_counts)
    
    print(f"\n{'='*70}")
    print(f"{split_name} Statistics")
    print(f"{'='*70}")
    print(f"Number of classes: {num_classes}")
    print(f"Total images: {total:,}")
    print(f"Average per class: {total/num_classes:.1f}")
    print(f"Min per class: {min(class_counts.values()):,}")
    print(f"Max per class: {max(class_counts.values()):,}")
    print(f"{'='*70}\n")


def get_gpu_info():
    gpus = tf.config.list_physical_devices('GPU')
    print(f"[INFO] TensorFlow version: {tf.__version__}")
    print(f"[INFO] GPU available: {len(gpus) > 0}")
    
    if len(gpus) > 0:
        for i, gpu in enumerate(gpus):
            print(f"[INFO] GPU {i}: {gpu}")
