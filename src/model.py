"""
Model architecture for QuickDraw Sketch Classification
CNN-based architecture with batch normalization and dropout
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from typing import Tuple, Dict, Any
import io


def build_quickdraw_cnn(
    input_shape: Tuple[int, int, int],
    num_classes: int,
    config: Dict[str, Any]
) -> keras.Model:
    l2_weight = config['model']['l2_weight']
    dropout = config['model']['dropout']
    dense_units = config['model']['dense_units']
    
    inputs = keras.Input(shape=input_shape)
    
    # Conv Block 1: 32 filters
    x = layers.Conv2D(32, 3, padding='same', 
                     kernel_regularizer=regularizers.l2(l2_weight))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(2)(x)
    
    # Conv Block 2: 64 filters (2 layers)
    x = layers.Conv2D(64, 3, padding='same',
                     kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    x = layers.Conv2D(64, 3, padding='same',
                     kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(dropout * 0.3)(x)
    
    # Conv Block 3: 128 filters (2 layers)
    x = layers.Conv2D(128, 3, padding='same',
                     kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    x = layers.Conv2D(128, 3, padding='same',
                     kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(dropout * 0.5)(x)
    
    # Conv Block 4: 256 filters (2 layers)
    x = layers.Conv2D(256, 3, padding='same',
                     kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    x = layers.Conv2D(256, 3, padding='same',
                     kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # Global pooling and dense layers
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    
    x = layers.Dense(dense_units, activation='relu',
                    kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = layers.Dropout(dropout * 0.5)(x)
    
    # Output layer
    x = layers.Dense(num_classes)(x)
    outputs = layers.Activation('softmax', dtype='float32')(x)
    
    model = keras.Model(inputs, outputs, name='quickdraw_cnn')
    
    return model


def compile_model(
    model: keras.Model,
    config: Dict[str, Any]
) -> keras.Model:
    # Optimizer
    optimizer_config = config['optimizer']
    optimizer = keras.optimizers.Adam(
        learning_rate=optimizer_config['learning_rate'],
        clipnorm=optimizer_config['clipnorm']
    )
    
    # Loss
    loss_config = config['loss']
    loss = keras.losses.CategoricalCrossentropy(
        label_smoothing=loss_config['label_smoothing'],
        from_logits=False
    )
    
    # Metrics
    top_k = config['evaluation']['top_k']
    metrics = [
        'accuracy',
        keras.metrics.TopKCategoricalAccuracy(k=top_k, name=f'top{top_k}_acc')
    ]
    
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )
    
    print(f"[INFO] Model compiled successfully")
    return model


def get_model_summary(model: keras.Model) -> str:
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + '\n'))
    return stream.getvalue()


def count_parameters(model: keras.Model) -> Dict[str, int]:
    trainable = sum([tf.size(w).numpy() for w in model.trainable_weights])
    non_trainable = sum([tf.size(w).numpy() for w in model.non_trainable_weights])
    
    return {
        'trainable': trainable,
        'non_trainable': non_trainable,
        'total': trainable + non_trainable
    }
