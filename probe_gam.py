'''
Return the value of the probe function, based on GAM model.
'''
import pyarrow as pa
import numpy as np
from coefficient import ProbeBase
from rpy2.robjects.packages import importr
import rpy2_arrow.arrow as pyra
from config import *

mgcv = importr('mgcv')
base = importr('base')
parallel = importr('parallel')
cl = parallel.makeCluster(8)

class Probe(ProbeBase):
    '''
    Return the value of the probe function, based on histogram.h5.
    '''

    probe = None

    def __init__(self, fn):
        '''
        Load the probe data from the fn.
        '''
        self.probe = base.readRDS(fn)
        
    def get_mu(self, rs, thetas):
        thetas[np.where(thetas > np.pi)] = 2*np.pi - thetas[np.where(thetas > np.pi)] 
        shape = rs.shape
        rs = rs.flatten()
        thetas = thetas.flatten()
        xy = pyra.converter.py2rpy(pa.Table.from_arrays([rs * np.cos(thetas), rs * np.sin(thetas)],
                                                        names=["x", "y"]))
        res = mgcv.predict_bam(self.probe.rx2("model_s"), base.as_data_frame(xy), newdata_guaranteed=True, cluster=cl)
        result = np.exp(res).reshape(shape)
        return result/FACTOR

    def get_lc(self, rs, thetas, ts):
        ts[ts > T_MAX] = T_MAX
        shape = rs.shape
        rs = rs.flatten()
        thetas = thetas.flatten()
        ts = ts.flatten()
        xyt = pyra.converter.py2rpy(pa.Table.from_arrays([rs * np.cos(thetas), rs * np.sin(thetas), ts],
                                                        names=["x", "y", "t"]))
        res = mgcv.predict_bam(self.probe.rx2("model_t"), base.as_data_frame(xyt), newdata_guaranteed=True, cluster=cl)
        result = np.exp(res).reshape(shape)
        return result/FACTOR

