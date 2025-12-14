"""
Training pipeline for QuickDraw Sketch Classification
Here there are callbacks setup, training loop, and history management
"""

import os
import time
import tensorflow as tf
from tensorflow import keras
from datetime import datetime
from typing import Dict, Any, Tuple, Optional
import numpy as np


def create_callbacks(config: Dict[str, Any]) -> list:
    callbacks = []
    
    # Model checkpoint
    checkpoint_config = config['callbacks']['model_checkpoint']
    checkpoint_path = os.path.join(
        config['output']['models_dir'],
        config['output']['checkpoint_name']
    )
    
    checkpoint = keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor=checkpoint_config['monitor'],
        save_best_only=checkpoint_config['save_best_only'],
        verbose=1
    )
    callbacks.append(checkpoint)
    
    # Early stopping
    es_config = config['callbacks']['early_stopping']
    early_stopping = keras.callbacks.EarlyStopping(
        monitor=es_config['monitor'],
        patience=es_config['patience'],
        restore_best_weights=es_config['restore_best_weights'],
        verbose=1
    )
    callbacks.append(early_stopping)
    
    # Reduce learning rate on plateau
    lr_config = config['callbacks']['reduce_lr']
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor=lr_config['monitor'],
        factor=lr_config['factor'],
        patience=lr_config['patience'],
        min_lr=lr_config['min_lr'],
        verbose=1
    )
    callbacks.append(reduce_lr)
    
    print(f"[INFO] Created {len(callbacks)} callbacks")
    return callbacks


def get_class_weights(
    class_counts: Dict[str, int],
    class_names: list,
    enabled: bool = False
) -> Optional[Dict[int, float]]:
    if not enabled:
        print("[INFO] Class weights disabled")
        return None
    
    try:
        counts_arr = np.array([class_counts.get(name, 1) for name in class_names])
        median = np.median(counts_arr)
        class_weights = {
            i: float(median / max(counts_arr[i], 1)) 
            for i in range(len(class_names))
        }
        print(f"[INFO] Class weights calculated")
        return class_weights
    except Exception as e:
        print(f"[WARNING] Could not calculate class weights: {e}")
        return None


def train_model(
    model: keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    config: Dict[str, Any],
    class_weights: Optional[Dict[int, float]] = None
) -> Tuple[keras.Model, Dict[str, list], float]:
    callbacks = create_callbacks(config)
    
    epochs = config['training']['epochs']
      
    start_time = time.time()
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    elapsed = time.time() - start_time
    
    print(f"\n[INFO] Training completed in {elapsed/60:.2f} minutes")
    print(f"{'='*70}\n")
    
    return model, history.history, elapsed


def save_model_and_history(
    model: keras.Model,
    history: Dict[str, list],
    config: Dict[str, Any],
    elapsed_time: float
):
    from src.utils import save_history, get_timestamp
    
    # Fin model
    model_path = os.path.join(
        config['output']['models_dir'],
        config['output']['final_model_name']
    )
    model.save(model_path)
    print(f"[INFO] Final model saved to {model_path}")
    
    # History
    timestamp = get_timestamp()
    history_path = os.path.join(
        config['output']['history_dir'],
        f'history_{timestamp}.pkl'
    )
    save_history(history, history_path)
    
    # Training summary
    summary_path = os.path.join(
        config['output']['logs_dir'],
        f'training_summary_{timestamp}.txt'
    )
    
    with open(summary_path, 'w') as f:
        f.write(f"Training Summary\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Duration: {elapsed_time/60:.2f} minutes\n")
        f.write(f"Epochs completed: {len(history['loss'])}\n")
        f.write(f"\nFinal Metrics:\n")
        f.write(f"Train Loss: {history['loss'][-1]:.4f}\n")
        f.write(f"Train Accuracy: {history['accuracy'][-1]:.4f}\n")
        f.write(f"Val Loss: {history['val_loss'][-1]:.4f}\n")
        f.write(f"Val Accuracy: {history['val_accuracy'][-1]:.4f}\n")
        f.write(f"\nBest Metrics:\n")
        f.write(f"Best Val Accuracy: {max(history['val_accuracy']):.4f}\n")
        f.write(f"Best Val Loss: {min(history['val_loss']):.4f}\n")
    
    print(f"[INFO] Training summary saved to {summary_path}")


def load_best_model(config: Dict[str, Any]) -> keras.Model:
    checkpoint_path = os.path.join(
        config['output']['models_dir'],
        config['output']['checkpoint_name']
    )
    
    if os.path.exists(checkpoint_path):
        model = keras.models.load_model(checkpoint_path)
        print(f"[INFO] Best model loaded from {checkpoint_path}")
        return model
    else:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
