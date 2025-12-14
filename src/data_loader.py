"""
Data loading and preprocessing for QuickDraw Sketch Classification
Here there are dataset preparation, loading, augmentation, and preprocessing
"""

import os
import shutil
import random
import tensorflow as tf
from tensorflow import keras
from typing import Dict, Tuple, Any


def prepare_dataset(
    source_dir: str,
    dest_dir: str,
    max_per_class: int,
    seed: int = 42
) -> Tuple[Dict[str, int], int]:
    if not os.path.isdir(source_dir):
        raise FileNotFoundError(f"Source directory '{source_dir}' not found")
    
    label_dirs = sorted([d for d in os.listdir(source_dir) 
                        if os.path.isdir(os.path.join(source_dir, d))])
    
    if not label_dirs:
        raise RuntimeError(f"No class subdirectories found in '{source_dir}'")
    
    if os.path.exists(dest_dir):
        try:
            shutil.rmtree(dest_dir)
            print(f"[INFO] Removed existing directory: {dest_dir}")
        except Exception as e:
            print(f"[WARNING] Could not remove directory: {e}")
    
    os.makedirs(dest_dir, exist_ok=True)
    
    for class_name in label_dirs:
        src_dir = os.path.join(source_dir, class_name)
        dst_dir = os.path.join(dest_dir, class_name)
        os.makedirs(dst_dir, exist_ok=True)
        
        imgs = [f for f in os.listdir(src_dir) 
               if f.lower().endswith('.png')]
        imgs = sorted(imgs)
        
        if len(imgs) == 0:
            print(f"[WARNING] No images found in {src_dir}")
            continue
        
        if len(imgs) <= max_per_class:
            chosen = imgs
        else:
            random.seed(seed)
            chosen = imgs.copy()
            random.shuffle(chosen)
            chosen = chosen[:max_per_class]
        
        for fname in chosen:
            src = os.path.join(src_dir, fname)
            dst = os.path.join(dst_dir, fname)
            
            if os.path.exists(dst):
                continue
            
            try:
                os.symlink(os.path.abspath(src), os.path.abspath(dst))
            except Exception:
                try:
                    shutil.copy2(src, dst)
                except Exception as e:
                    print(f"[WARNING] Could not copy {fname}: {e}")
    
    from src.utils import count_images
    class_counts, total_images = count_images(dest_dir)
    
    print(f"[INFO] Dataset prepared: {dest_dir}")
    print(f"[INFO] Classes: {len(class_counts)}, Total images: {total_images:,}")
    
    return class_counts, total_images


def load_datasets(
    data_dir: str,
    img_size: Tuple[int, int],
    batch_size: int,
    validation_split: float,
    seed: int = 42
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, list]:
    print("[INFO] Loading datasets...")
    
    raw_train = keras.utils.image_dataset_from_directory(
        data_dir,
        labels='inferred',
        label_mode='categorical',
        batch_size=batch_size,
        image_size=img_size,
        validation_split=validation_split,
        subset='training',
        seed=seed,
        shuffle=True
    )
    
    raw_val_test = keras.utils.image_dataset_from_directory(
        data_dir,
        labels='inferred',
        label_mode='categorical',
        batch_size=batch_size,
        image_size=img_size,
        validation_split=validation_split,
        subset='validation',
        seed=seed,
        shuffle=True  
    )
    
    class_names = raw_train.class_names
    num_classes = len(class_names)
    
    val_test_steps = int(tf.data.experimental.cardinality(raw_val_test).numpy())
    val_steps = val_test_steps // 2
    test_steps = val_test_steps - val_steps
    
    raw_val = raw_val_test.take(val_steps)
    raw_test = raw_val_test.skip(val_steps)
    
    train_steps = int(tf.data.experimental.cardinality(raw_train).numpy())
    
    print(f"[INFO] Classes: {num_classes}")
    print(f"[INFO] Train batches: {train_steps}, Val batches: {val_steps}, Test batches: {test_steps}")
    
    return raw_train, raw_val, raw_test, class_names


def augment_batch(images, labels):
    images = tf.cast(images, tf.float32)
    labels = tf.cast(labels, tf.float32)
    
    images = images / 255.0
    
    images = tf.image.random_flip_left_right(images)
    images = tf.image.random_brightness(images, 0.1)
    images = tf.image.random_contrast(images, 0.9, 1.1)
    
    images = tf.clip_by_value(images, 0.0, 1.0)
    
    return images, labels


def preprocess_batch(images, labels):
    images = tf.cast(images, tf.float32) / 255.0
    labels = tf.cast(labels, tf.float32)
    return images, labels


def apply_mixup(images, labels, alpha: float = 0.2):
    batch_size = tf.shape(images)[0]
    
    # Sample lambda from Beta distribution
    d1 = tf.random.gamma([batch_size], alpha, 1.0)
    d2 = tf.random.gamma([batch_size], alpha, 1.0)
    lam = tf.cast(d1 / (d1 + d2), tf.float32)
    
    # Reshape for broadcasting
    lam_img = tf.reshape(lam, [batch_size, 1, 1, 1])
    lam_lbl = tf.reshape(lam, [batch_size, 1])
    
    # Shuffle indices
    idx = tf.random.shuffle(tf.range(batch_size))
    
    # Mix images and labels
    mixed_images = lam_img * images + (1 - lam_img) * tf.gather(images, idx)
    mixed_labels = lam_lbl * labels + (1 - lam_lbl) * tf.gather(labels, idx)
    
    return mixed_images, mixed_labels


def create_datasets(
    raw_train: tf.data.Dataset,
    raw_val: tf.data.Dataset,
    raw_test: tf.data.Dataset,
    config: Dict[str, Any]
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    AUTOTUNE = tf.data.AUTOTUNE
    
    train_ds = raw_train.map(augment_batch, num_parallel_calls=AUTOTUNE)
    
    mixup_alpha = config['augmentation']['mixup_alpha']
    if mixup_alpha > 0:
        train_ds = train_ds.map(
            lambda x, y: apply_mixup(x, y, mixup_alpha),
            num_parallel_calls=AUTOTUNE
        )
        print(f"[INFO] Mixup augmentation enabled (alpha={mixup_alpha})")
    
    train_ds = train_ds.prefetch(AUTOTUNE)
    
    val_ds = raw_val.map(preprocess_batch, num_parallel_calls=AUTOTUNE).cache().prefetch(AUTOTUNE)
    test_ds = raw_test.map(preprocess_batch, num_parallel_calls=AUTOTUNE).cache().prefetch(AUTOTUNE)
    
    print("[INFO] Datasets created with preprocessing and augmentation")
    
    return train_ds, val_ds, test_ds
