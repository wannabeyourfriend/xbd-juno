'''
Return the value of the probe function, based on GBM model.
'''
import numpy as np
import pickle
from coefficient import ProbeBase
from config import *

class Probe(ProbeBase):
    def __init__(self, fn):
        with open(fn, "rb") as fin:
            models = pickle.load(fin)
        self.model_s = models["model_s"]
        self.model_t = models["model_t"]
        self.t_binwidth = models["t_binwidth"]

        # 添加边缘修正系数
        #self.edge_corr = lambda r: 1 + 0.3*np.exp(-6*(1-r))

    def get_mu(self, rs, thetas):
        thetas[np.where(thetas > np.pi)] = 2*np.pi - thetas[np.where(thetas > np.pi)]
        shape = rs.shape
        x = (rs * np.cos(thetas)).flatten()
        y = (rs * np.sin(thetas)).flatten()
        r = rs.flatten()
        log_nEV = np.zeros_like(x)
        X = np.stack([x, y, r, r**2, log_nEV], axis=1)
        pred = self.model_s.predict(X) #* self.edge_corr(r)
        result = pred.reshape(shape)  

        return result / FACTOR

    def get_lc(self, rs, thetas, ts):
        shape = rs.shape
        x = (rs * np.cos(thetas)).flatten()
        y = (rs * np.sin(thetas)).flatten()
        r = rs.flatten()
        t = ts.flatten()
        log_nEV = np.zeros_like(x)
        X = np.stack([x, y, r, r**2, t, log_nEV], axis=1)  
        pred = self.model_t.predict(X) #* self.edge_corr(r)
        result = pred.reshape(shape)
        return result / FACTOR / self.t_binwidth 
