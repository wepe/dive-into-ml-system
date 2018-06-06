from lr import model
import numpy as np


features = np.load('features.dat')
labels = np.load('labels.dat')


clf = model(max_iter=500,alpha=0.1,l2_lambda=0.01,tolerance=0.001)

clf.fit(features,labels)
print labels[:30]
print clf.predict(features[:30])

