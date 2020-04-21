from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import numpy as np

def loss_fft(y_true, y_pred):        
    fft_true = np.abs(np.fft.rfft(y_true))
    fft_pred = np.abs(np.fft.rfft(y_pred))
    loss = np.mean(np.square(np.subtract(fft_true,fft_pred)))
    return loss, fft_true, fft_pred

def cross_correlation(y_true, y_pred):
    cc = np.correlate(y_true,y_pred)
    return cc

def dtw_distance(y_true, y_pred):        
    #distance = dtw.distance(y_true, y_pred)
    distance, path = fastdtw(y_true, y_pred, dist=euclidean)
    return distance