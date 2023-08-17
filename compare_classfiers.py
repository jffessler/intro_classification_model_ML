from sklearn.datasets import make_classification

X,Y = make_classification(n_samples=1000,n_classes=2,n_features=5,n_redundant=0, random_state=1)

# print(X.shape)
# print(Y.shape)

#split the data into 80/20
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2)
# print(X_train.shape, Y_train.shape)
# print(X_test.shape,Y_test.shape)

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import SGDClassifier

names = ["Nearest_Neighbors", "Linear_SVM", "Polynomial_SVM", "RBF_SVM", "Gaussian_Process",
         "Gradient_Boosting", "Decision_Tree", "Extra_Trees", "Random_Forest", "Neural_Net", "AdaBoost",
         "Naive_Bayes", "QDA", "SGD"]

#all the classifers to be tested
classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(kernel="poly", degree=3, C=0.025),
    SVC(kernel="rbf", C=1, gamma=2),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    GradientBoostingClassifier(n_estimators=100, learning_rate=1.0),
    DecisionTreeClassifier(max_depth=5),
    ExtraTreesClassifier(n_estimators=10, min_samples_split=2),
    RandomForestClassifier(max_depth=5, n_estimators=100),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(n_estimators=100),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
    SGDClassifier(loss="hinge", penalty="l2")]

scores =[]
for name, clf in zip(names,classifiers):
    clf.fit(X_train,Y_train)
    score = clf.score(X_test,Y_test)
    scores.append(score)

print(scores)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display


df = pd.DataFrame()
df['name'] = names
df['score'] = scores
# df.plot()
plt.show()

cm = sns.light_palette("green",as_cmap=True)
s = df.style.background_gradient(cmap=cm)
display(s)

# sns.set(style="whitegrid")
# ax = sns.barplot(y="name",x="score",data=df)
# plt.show()