from lr import model
import numpy as np

features = np.load('features.dat')
labels = np.load('labels.dat')

clf = model(max_iter=10,alpha=0.1,l2_lambda=0.01,tolerance=0.001)
clf.fit(features,labels)
print clf.predict(features[:30])

clf.save("/home/wepon/lr.model")
clf1 = model()
clf1.load("/home/wepon/lr.model")
print clf1.predict(features[:30])
