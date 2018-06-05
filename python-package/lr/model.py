from ctypes import *
import numpy as np
import os

liblr = cdll.LoadLibrary(os.path.dirname(os.path.realpath(__file__))+'/liblr.so')

class model(object):
    def __init__(self,max_iter=200,alpha=0.01,l2_lambda=0.01,tolerance=0.001):
        self.max_iter = max_iter
        self.alpha = alpha
        self.l2_lambda = l2_lambda
        self.tolerance = tolerance
        self.fmodel = None

    # only support python list
    def _fit(self,features,labels):
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
        liblr.fit.argtypes = [POINTER(POINTER(c_double)),POINTER(c_int),c_int,c_int,c_int,c_double,c_double,c_double]
        liblr.fit.restype = POINTER(c_char)
        res = liblr.fit(double_p_p,int_p,c_int(row),c_int(col),c_int(self.max_iter),c_double(self.alpha),c_double(self.l2_lambda),c_double(self.tolerance))
        self.fmodel = ''.join([res[i] for i in range(25)])

    # only support python list
    def _predict_prob(self,features):
        assert self.fmodel is not None
        features = [tuple(f) for f in features]
        row = len(features)
        col = len(features[0])
        # initialize ctypes array
        double_2d_array = ((c_double*col)*row)(*features)
        del features
        # cast to C function argument's type
        double_p_list = []
        for i in range(row):
            double_p_list.append(cast(double_2d_array[i],POINTER(c_double)))
        double_p_p = (POINTER(c_double)*row)(*double_p_list)
        # call the C function
        liblr.predict_prob.argtypes = [POINTER(POINTER(c_double)),c_int,c_int,POINTER(c_char)]
        liblr.predict_prob.restype = POINTER(c_double)

        res = liblr.predict_prob(double_p_p,c_int(row),c_int(col),c_char_p(self.fmodel))
        return [res[i] for i in range(row)]


    # support python list, numpy array
    def fit(self,features,labels):
        # convert to numpy array
        if not isinstance(features,np.ndarray):
            features = np.array(features,dtype=np.double)
        if isinstance(labels,np.ndarray):
            labels = list(labels)

        # convert to ctypes's type
        row,col = features.shape
        int_p = cast((c_int*row)(*labels),POINTER(c_int))
        double_p_p = (features.ctypes.data + np.arange(features.shape[0]) * features.strides[0]).astype(np.uintp)

        # call the C function
        DOUBLEPP = np.ctypeslib.ndpointer(dtype=np.uintp,ndim=1,flags='C')
        INTP = POINTER(c_int)
        liblr.fit.argtypes = [DOUBLEPP,INTP,c_int,c_int,c_int,c_double,c_double,c_double]
        liblr.fit.restype = POINTER(c_char)
        res = liblr.fit(double_p_p,int_p,c_int(row),c_int(col),c_int(self.max_iter),c_double(self.alpha),c_double(self.l2_lambda),c_double(self.tolerance))
        self.fmodel = ''.join([res[i] for i in range(25)])


    def predict_prob(self,features):
        assert self.fmodel is not None
        # convert to numpy array
        if not isinstance(features,np.ndarray):
            features = np.array(features,dtype=np.double)

        # convert to ctypes's type
        row,col = features.shape
        double_p_p = (features.ctypes.data + np.arange(features.shape[0]) * features.strides[0]).astype(np.uintp)

        # call C function
        DOUBLEPP = np.ctypeslib.ndpointer(dtype=np.uintp,ndim=1,flags='C')
        liblr.predict_prob.argtypes = [DOUBLEPP,c_int,c_int,c_char_p]
        liblr.predict_prob.restype = POINTER(c_double)
        res = liblr.predict_prob(double_p_p,c_int(row),c_int(col),c_char_p(self.fmodel))
        return [res[i] for i in range(row)]

    def predict(self,features):
        assert self.fmodel is not None
        prob = self.predict_prob(features)
        return [1 if p>0.5 else 0 for p in prob]

