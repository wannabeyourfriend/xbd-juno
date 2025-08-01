import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras
from coefficient import ProbeBase
from config import *

# Replace this line
# @keras.saving.register_keras_serializable()
# With this
@tf.keras.utils.register_keras_serializable()
def custom_poisson_loss(y_true, y_pred):
    y_pred = tf.maximum(y_pred, 1e-8)
    return tf.reduce_mean(y_pred - y_true * tf.math.log(y_pred))

class Probe(ProbeBase):
    
    def __init__(self, fn):
        with open(fn, "rb") as f:
            models_data = pickle.load(f)
        
        self.model_s = models_data["model_s"]
        self.model_t = models_data["model_t"]
        self.scaler_s = models_data["scaler_s"]
        self.scaler_t = models_data["scaler_t"]
        self.t_binwidth = models_data["t_binwidth"]
        
        self.model_s.trainable = False
        self.model_t.trainable = False
    
    def _prepare_spatial_features(self, rs, thetas):
        thetas = np.where(thetas > np.pi, 2*np.pi - thetas, thetas)
        
        x = rs * np.cos(thetas)
        y = rs * np.sin(thetas)
        r = rs
        
        r2 = r**2
        r3 = r**3
        theta = np.arctan2(y, x)
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        log_nEV = np.zeros_like(x) 
        
        xy = x * y
        xr = x * r
        yr = y * r
        
        features = np.stack([
            x, y, r, r2, r3,
            theta, sin_theta, cos_theta,
            log_nEV, xy, xr, yr
        ], axis=-1)
        
        return features
    
    def _prepare_temporal_features(self, rs, thetas, ts):
        thetas = np.where(thetas > np.pi, 2*np.pi - thetas, thetas)
        
        ts = np.clip(ts, 0, T_MAX)
        
        x = rs * np.cos(thetas)
        y = rs * np.sin(thetas)
        r = rs
        
        r2 = r**2
        r3 = r**3
        theta = np.arctan2(y, x)
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        t_norm = ts / T_MAX
        log_t = np.log(ts + 1)
        log_nEV = np.zeros_like(x)
        
        xy = x * y
        xr = x * r
        yr = y * r
        rt = r * ts
        
        features = np.stack([
            x, y, r, r2, r3,
            theta, sin_theta, cos_theta,
            ts, t_norm, log_t,
            log_nEV, xy, xr, yr, rt
        ], axis=-1)
        
        return features
    
    def get_mu(self, rs, thetas):
        original_shape = rs.shape
        
        rs_flat = rs.flatten()
        thetas_flat = thetas.flatten()
        
        features = self._prepare_spatial_features(rs_flat, thetas_flat)
        
        features_scaled = self.scaler_s.transform(features)
        
        predictions = self.model_s.predict(features_scaled, verbose=0)
        
        result = predictions.reshape(original_shape)
        
        return result / FACTOR
    
    def get_lc(self, rs, thetas, ts):
        original_shape = rs.shape
        
        rs_flat = rs.flatten()
        thetas_flat = thetas.flatten()
        ts_flat = ts.flatten()
        
        features = self._prepare_temporal_features(rs_flat, thetas_flat, ts_flat)
        
        features_scaled = self.scaler_t.transform(features)
        
        predictions = self.model_t.predict(features_scaled, verbose=0)
        
        result = predictions.reshape(original_shape)
        
        # Remove the t_binwidth division to match GAM approach
        return result / FACTOR
    
    def get_pie(self, rs, thetas):
        thetas2, rs2 = np.meshgrid(thetas, rs)
        return self.get_mu(rs2, thetas2)