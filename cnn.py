#!/usr/bin/env python3

import argparse
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pickle
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from config import *

tf.config.run_functions_eagerly(True)

tf.random.set_seed(42)
np.random.seed(42)

def create_spatial_cnn_model(input_shape):
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.1),
        
        layers.Dense(1, activation='exponential')
    ])
    
    return model

def create_temporal_cnn_model(input_shape):
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(1024, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.1),
        
        layers.Dense(1, activation='exponential')
    ])
    
    return model

def preprocess_spatial_data(table_s):
    x_s = table_s["x"].to_numpy()
    y_s = table_s["y"].to_numpy()
    r_s = np.sqrt(x_s**2 + y_s**2)
    nEV_s = table_s["nEV"].to_numpy()
    nPE_s = table_s["nPE"].to_numpy()
    
    log_nEV_s = np.log(nEV_s + 1e-8)
    
    theta_s = np.arctan2(y_s, x_s)
    r2_s = r_s**2
    r3_s = r_s**3
    
    X_s = np.stack([
        x_s, y_s, r_s, r2_s, r3_s,
        theta_s, np.sin(theta_s), np.cos(theta_s),
        log_nEV_s, x_s*y_s, x_s*r_s, y_s*r_s
    ], axis=1)
    
    return X_s, nPE_s

def preprocess_temporal_data(table_t):
    x_t = table_t["x"].to_numpy()
    y_t = table_t["y"].to_numpy()
    r_t = np.sqrt(x_t**2 + y_t**2)
    t_t = table_t["t"].to_numpy()
    nEV_t = table_t["nEV"].to_numpy()
    nPE_t = table_t["nPE"].to_numpy()
    
    log_nEV_t = np.log(nEV_t + 1e-8)
    
    theta_t = np.arctan2(y_t, x_t)
    r2_t = r_t**2
    r3_t = r_t**3
    t_norm = t_t / T_MAX 
    
    X_t = np.stack([
        x_t, y_t, r_t, r2_t, r3_t,
        theta_t, np.sin(theta_t), np.cos(theta_t),
        t_t, t_norm, np.log(t_t + 1),
        log_nEV_t, x_t*y_t, x_t*r_t, y_t*r_t, r_t*t_t
    ], axis=1)
    
    return X_t, nPE_t

# Replace this line
# @keras.saving.register_keras_serializable()
# With this
@tf.keras.utils.register_keras_serializable()
def custom_poisson_loss(y_true, y_pred):
    y_pred = tf.maximum(y_pred, 1e-8)
    return tf.reduce_mean(y_pred - y_true * tf.math.log(y_pred))

def main():
    parser = argparse.ArgumentParser(description="CNN训练脚本")
    parser.add_argument("-i", dest="ipt", nargs="+", type=str, help="输入parquet文件")
    parser.add_argument("-o", dest="opt", type=str, help="输出模型文件")
    parser.add_argument("--epochs", type=int, default=200, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=1024, help="批次大小")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="学习率")
    args = parser.parse_args()
    
    print("开始训练CNN模型...")
    
    print("读取空间数据...")
    table_s = pq.read_table(args.ipt[0])
    X_s, y_s = preprocess_spatial_data(table_s)
    
    print("读取时空数据...")
    table_t = pq.read_table(args.ipt[1])
    X_t, y_t = preprocess_temporal_data(table_t)
    
    print("数据标准化...")
    scaler_s = StandardScaler()
    X_s_scaled = scaler_s.fit_transform(X_s)
    
    scaler_t = StandardScaler()
    X_t_scaled = scaler_t.fit_transform(X_t)
    
    X_s_train, X_s_val, y_s_train, y_s_val = train_test_split(
        X_s_scaled, y_s, test_size=0.2, random_state=42
    )
    
    X_t_train, X_t_val, y_t_train, y_t_val = train_test_split(
        X_t_scaled, y_t, test_size=0.2, random_state=42
    )
    
    print("创建空间CNN模型...")
    model_s = create_spatial_cnn_model((X_s_scaled.shape[1],))
    
    print("创建时空CNN模型...")
    model_t = create_temporal_cnn_model((X_t_scaled.shape[1],))
    
    optimizer_s = keras.optimizers.Adam(learning_rate=args.learning_rate)
    optimizer_t = keras.optimizers.Adam(learning_rate=args.learning_rate)
    
    model_s.compile(
        optimizer=optimizer_s,
        loss=custom_poisson_loss,
        metrics=['mae', 'mse']
    )
    
    model_t.compile(
        optimizer=optimizer_t,
        loss=custom_poisson_loss,
        metrics=['mae', 'mse']
    )
    
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-6
        )
    ]
    
    print("训练空间模型...")
    history_s = model_s.fit(
        X_s_train, y_s_train,
        validation_data=(X_s_val, y_s_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    print("训练时空模型...")
    history_t = model_t.fit(
        X_t_train, y_t_train,
        validation_data=(X_t_val, y_t_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    t_bins = np.unique(table_t["t"].to_numpy())
    t_binwidth = t_bins[1] - t_bins[0] if len(t_bins) > 1 else T_MAX / 30
    
    print(f"保存模型到 {args.opt}")
    models_data = {
        "model_s": model_s,
        "model_t": model_t,
        "scaler_s": scaler_s,
        "scaler_t": scaler_t,
        "t_binwidth": t_binwidth,
        "history_s": history_s.history,
        "history_t": history_t.history
    }
    
    with open(args.opt, "wb") as f:
        pickle.dump(models_data, f)
    
    print("CNN模型训练完成!")
    print(f"空间模型最终验证损失: {min(history_s.history['val_loss']):.6f}")
    print(f"时空模型最终验证损失: {min(history_t.history['val_loss']):.6f}")

if __name__ == "__main__":
    main()