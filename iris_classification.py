from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

iris = datasets.load_iris()
# print(iris.feature_names)
# print(iris.target_names)

x = iris.data
y = iris.target
# print(x.shape)
# print(y.shape)

clf = RandomForestClassifier()
# clf.fit(x,y)

# print(clf.feature_importances_)
# print(clf.predict([x[0]]))
# print(clf.predict_proba([x[0]]))

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

# print(x_train.shape)
# print(x_test.shape)
# print(y_train.shape,y_test.shape)

clf.fit(x_train,y_train)

# print(clf.predict([x[0]]))
# print(clf.predict_proba([x[0]]))
print(clf.predict(x_test))
print(y_test)
print(clf.score(x_test,y_test))
