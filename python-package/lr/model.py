from ctypes import *
import os

liblr = cdll.LoadLibrary('./liblr.so')

class model(object):
    def __init__(self,max_iter=200,alpha=0.01,l2_lambda=0.01,tolerance=0.001):
        self.max_iter = max_iter
        self.alpha = alpha
        self.l2_lambda = l2_lambda
        self.tolerance = tolerance
        self.fmodel = None

    # TO support numpy array
    def fit(self,features,labels):
        features = [tuple(f) for f in features]
        row = len(features)
        col = len(features[0])
        # initialize ctypes array
        double_2d_array = ((c_double*col)*row)(*features)
        int_array = (c_int*row)(*labels)
        del features,labels
        # cast to C function argument's type
        int_p = cast(int_array,POINTER(c_int))
        double_p_list = []
        for i in range(row):
            double_p_list.append(cast(double_2d_array[i],POINTER(c_double)))
        double_p_p = (POINTER(c_double)*row)(*double_p_list)
        # call the C function
        liblr.train.argtypes = [POINTER(POINTER(c_double)),POINTER(c_int),c_int,c_int]
        liblr.train.restype = POINTER(c_double)
        res = liblr.train(double_p_p,int_p,c_int(row),c_int(col))

    def predict_prob(self,features):
        pass

    def predict(self,features):
        pass
