from sklearn.datasets import make_classification
import pandas as pd
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import numpy as np

X,Y = make_classification(n_samples=200,n_classes=2,n_features=10,n_redundant=0, random_state=1)

x = pd.DataFrame(X)
# display(x)

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2)
# print(X_train.shape, Y_train.shape)
# print(X_test.shape,Y_test.shape)

#setting up machine learning model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

rf = RandomForestClassifier(max_features=5, n_estimators=100)

#basic score calculation
rf.fit(X_train,Y_train)
# print(rf.score(X_test,Y_test))

#more complex score calculation, however accuracy score and score always give the same answer but with different approaches

yPredict = rf.predict(X_test)
# print(accuracy_score(yPredict,Y_test))

#hyper parameter tuning

max_feat_range = np.arange(1,6,1)
n_esti_range = np.arange(10,210,10)
param_grid = dict(max_features = max_feat_range, n_estimators = n_esti_range)

rf = RandomForestClassifier()

grid = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5)

z = grid.fit(X_train,Y_train)
# print(z)
print(f"Best Score: {z.best_params_} with the best score of: {z.best_score_}")

grid_results = pd.concat([pd.DataFrame(grid.cv_results_["params"]),pd.DataFrame(grid.cv_results_["mean_test_score"], columns=["Accuracy"])],axis=1)
grid_results.head()
# display(grid_results)

grid_contour = grid_results.groupby(["max_features","n_estimators"]).mean()
# display(grid_contour)

grid_reset = grid_contour.reset_index()
# display(grid_reset)
grid_reset.columns = ["max_features","n_estimators","Accuracy"]
grid_pivot = grid_reset.pivot(index="max_features",columns='n_estimators')
# display(grid_pivot)

x = grid_pivot.columns.levels[1].values
y = grid_pivot.index.values
z = grid_pivot.values

# print(f"x variable {x}")
# print(f"y variable {y}")
# print(f"z variable {z}")

#Plotting a 2D contour plot 
import plotly.graph_objects as go

# X and Y axes labels
layout = go.Layout(
            xaxis=go.layout.XAxis(
              title=go.layout.xaxis.Title(
              text='n_estimators')
             ),
             yaxis=go.layout.YAxis(
              title=go.layout.yaxis.Title(
              text='max_features') 
            ) )

fig = go.Figure(data = [go.Contour(z=z, x=x, y=y)], layout=layout )

fig.update_layout(title='Hyperparameter tuning', autosize=False,
                  width=500, height=500,
                  margin=dict(l=65, r=50, b=65, t=90))

fig.show()

# plotting 3D surface plot

fig = go.Figure(data= [go.Surface(z=z, y=y, x=x)], layout=layout )
fig.update_layout(title='Hyperparameter tuning',
                  scene = dict(
                    xaxis_title='n_estimators',
                    yaxis_title='max_features',
                    zaxis_title='Accuracy'),
                  autosize=False,
                  width=800, height=800,
                  margin=dict(l=65, r=50, b=65, t=90))
fig.show()