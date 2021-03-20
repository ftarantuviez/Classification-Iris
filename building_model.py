from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import pickle

iris = datasets.load_iris()
X = iris.data
Y = iris.target

clf = RandomForestClassifier()
clf.fit(X, Y)

# saving model
pickle.dump(clf, open('iris_clf.pkl', 'wb'))