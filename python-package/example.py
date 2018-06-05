from lr import model


features = [[1.0,0.8],[2.0,1.7],[3.0,2.5],[4.0,3.6],[5.0,4.9],[1.0,1.2],[2.0,2.5],[3.0,3.4],[4.0,4.5],[5.0,6.0]]
labels = [0,0,0,0,0,1,1,1,1,1]

clf = model(max_iter=200,alpha=0.01,l2_lambda=0.05,tolerance=0.001)

clf._fit(features,labels)
print clf._predict_prob(features)
print clf.predict(features)

