from lr import model
import numpy as np

# custom metric function, mean accuracy
def mean_accuracy(label,pred,size):
    num_pos,hit_pos = 0.0,0.0
    num_neg,hit_neg = 0.0,0.0
    for i in range(size):
        if label[i]==1.0:
            num_pos += 1.0
            if pred[i]>0.5:
                hit_pos += 1.0

        if label[i]==0.0:
            num_neg += 1.0
            if pred[i]<=0.5:
                hit_neg += 1.0
    print "pos-accracy:{0:.5f},neg-accuracy:{1:.5f}".format(hit_pos/num_pos,hit_neg/num_neg)
    return 0.5*hit_pos/num_pos + 0.5*hit_neg/num_neg


features = np.load('features.dat')
labels = np.load('labels.dat')
print features.shape,labels.shape,labels.sum()

clf = model(max_iter=1000,alpha=0.01,l2_lambda=0.5,tolerance=0.01)
clf.fit(features,labels,batch_size=1024,early_stopping_round=100,metric=mean_accuracy)
print clf.predict(features[:30])

clf.save("/home/wepon/lr.model")
clf1 = model()
clf1.load("/home/wepon/lr.model")
print clf1.predict(features[:30])

