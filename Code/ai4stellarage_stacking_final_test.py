import numpy as np
from numpy import mean
from numpy import std
from sklearn.datasets import make_regression
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import StackingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF # define the kernels to use in the GPs
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV #To perform the search for the best parameters
from sklearn.metrics import mean_absolute_error #define the metrics to use for the evaluation
from matplotlib import pyplot
from joblib import dump, load
import pandas as pd

# load model (and test data) for further analysis
models = load('results/models.joblib')
X_train, y_train, X_test_no, y_test_no = load('results/test_data.joblib')

# the script imports the data from a data file with the information of the stars
data = pd.read_csv('data/test_gyro.txt', sep=",", header=0)
#clean NA values
df = data[['m', 'r', 'teff','L','Meta','logg','prot','ager']]
df.dropna(inplace=True, axis=0)
#sort the dataframe by age
df = df.sort_values(by=['ager'])
#chose target variable: age
y_test_new = np.array(df['ager'])
#selection of the data to be used
X = np.array(df[['m', 'r', 'teff','L','Meta','logg','prot']])
#normalize X
X_test = preprocessing.scale(X)

##### data normalization
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test_new = scaler.transform(X_test)

'''
# array to a df to sort by age
aux = pd.DataFrame(X_test)
aux['ager'] = y_test
aux = aux.sort_values(by='ager')

# df to array again
X_test_new = aux.iloc[:,0:7]
y_test_new = aux['ager'].to_numpy()
'''

# select the model to use for the prediction
# 'lr'-> LinearRegression
# 'dtr' -> DecisionTreeRegressor
# 'rf' ->RandomForestRegressor
# 'svm'->SVR
# 'bayes'->BayesianRidge
# 'knn'->KNeighborsRegressor
# 'gp'->GaussianProcessRegressor
# 'nnet'->MLPRegressor
# 'stacking'->StackingRegressor

y_pred, y_std = models['stacking'].predict(X_test_new, return_std =True)
#y_pred = models['nnet'].predict(X_test_new)
#y_std = std(y_pred)
#Compare predictions with ground truth
score = mean_absolute_error(y_test_new, y_pred)
print("MAE: ", score)

reg_error = y_test_new-y_pred

df_final_test = pd.DataFrame(list(zip(y_test_new, y_pred, reg_error)),
                             columns =['y_test', 'y_pred', 'reg_error'])
df_final_test.to_csv('df_final_test_stacking.csv')

n = np.arange(y_pred.size)

fig, ax = pyplot.subplots()
ax.scatter(n, y_test_new, c='tab:blue', label='y_test',alpha=0.3, edgecolors='none')
ax.scatter(n, y_pred, c='tab:orange', label='y_pred',alpha=0.3, edgecolors='none')
ax.legend()
ax.grid(True)
pyplot.show()

fig, ax = pyplot.subplots()
ax.plot(n, y_test_new, c='tab:blue', label='y_test')
ax.plot(n, y_pred, c='tab:orange', label='y_pred')
ax.plot(n, reg_error, c='tab:red', label='error')
ax.fill_between(n, y_pred-y_std, y_pred+y_std,
                color="pink", alpha=0.5, label="std")

ax.legend()
ax.grid(True)
pyplot.show()
