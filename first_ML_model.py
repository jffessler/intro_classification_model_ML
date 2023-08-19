##First ML Project
import pandas as pd
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import numpy as np

#load in data set
df = pd.read_csv("https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv")

#data separation and preparation
y = df["logS"]
X = df.drop("logS",axis=1)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=100)

#build the model
##linear regression model
lr = LinearRegression()
lr.fit(X_train,y_train)

#training the model
y_lr_train_pred = lr.predict(X_train)
y_lr_test_pred = lr.predict(X_test)

#evaluate model predictions y_train/y_lr_train_pred
lr_train_mse = mean_squared_error(y_train,y_lr_train_pred)
lr_train_r2 = r2_score(y_train, y_lr_train_pred)
lr_test_mse = mean_squared_error(y_test, y_lr_test_pred)
lr_test_r2 = r2_score(y_test,y_lr_test_pred)

## results of linear regression model
# print(f"LR MSE (Train): {lr_train_mse} \nLR R2 (Train): {lr_train_r2} \nLR MSE (Test): {lr_test_mse} \nLR R2 (Test): {lr_test_r2}")
lr_results = pd.DataFrame(["Linear regression", lr_train_mse, lr_train_r2, lr_test_mse, lr_test_r2]).transpose()
lr_results.columns = ["Method", "Training MSE", "Training R2", "Test MSE", "Test R2"]
# print(lr_results)

### note: if the y variable of the data set is quantitative then use a Regression model
### while when the y variable is categorical, use a Classification model

# Random forest regression model build and train
rf = RandomForestRegressor(max_depth=2, random_state=100)
rf.fit(X_train,y_train)

#apply the model, make prediction
y_rf_train_pred = rf.predict(X_train)
y_rf_test_pred = rf.predict(X_test)

#evaluate performance of forest model
rf_train_mse = mean_squared_error(y_train,y_rf_train_pred)
rf_train_r2 = r2_score(y_train, y_rf_train_pred)
rf_test_mse = mean_squared_error(y_test, y_rf_test_pred)
rf_test_r2 = r2_score(y_test,y_rf_test_pred)
#results
rf_results = pd.DataFrame(["Random forest", rf_train_mse, rf_train_r2, rf_test_mse, rf_test_r2]).transpose()
rf_results.columns = ["Method", "Training MSE", "Training R2", "Test MSE", "Test R2"]
# print(rf_results)

## compare linear regressio and random forest models
df_models = pd.concat([lr_results,rf_results], axis=0).reset_index(drop=True)
print(df_models)

## Data visualization
plt.figure(figsize=(5,5))
plt.scatter(x=y_train,y=y_lr_train_pred, c="#7CAE00", alpha=0.3)

##trend line
z = np.polyfit(y_train,y_lr_train_pred,1)
p = np.poly1d(z)

plt.plot(y_train,p(y_train), '#F8766D')
plt.ylabel("Predict LogS")
plt.xlabel("Experimental LogS")
plt.show()
