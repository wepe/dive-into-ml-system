from ctypes import *

lr = cdll.LoadLibrary('./liblr.so')

lr.test_func()


def train(features,labels):
    features = [tuple(f) for f in features]
    row = len(features)
    col = len(features[0])
    double_2d_array = ((c_double*col)*row)(*features)
    int_array = (c_int*row)(*labels)

    int_p = cast(int_array,POINTER(c_int))
    double_p_list = []
    for i in range(row):
        double_p_list.append(cast(double_2d_array[i],POINTER(c_double)))
    double_p_p = (POINTER(c_double)*row)(*double_p_list)

    lr.train.argtypes = [POINTER(POINTER(c_double)),POINTER(c_int),c_int,c_int]
    lr.train.restype = POINTER(c_double)
    res = lr.train(double_p_p,int_p,c_int(row),c_int(col))
    print res[0],res[1],res[2]


features = [[1.0,0.8],[2.0,1.7],[3.0,2.5],[4.0,3.6],[5.0,4.9],[1.0,1.2],[2.0,2.5],[3.0,3.4],[4.0,4.5],[5.0,6.0]]
labels = [0,0,0,0,0,1,1,1,1,1]
train(features,labels)
