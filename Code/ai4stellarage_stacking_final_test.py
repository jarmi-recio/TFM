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
X_test, y_test = load('results/test_data.joblib')

# array to a df to sort by age
aux = pd.DataFrame(X_test)
aux['age'] = y_test
aux = aux.sort_values(by='age')

# df to array again
X_test_new = aux.iloc[:,0:8]
y_test_new = aux['age'].to_numpy()

# select the model to use for the prediction
# 'lr'-> LinearRegression
# 'dtr' -> DecisionTreeRegressor
# 'rf' ->RandomForestRegressor
# 'svm'->SVR
# 'bayes'->BayesianRidge
# 'knn'->KNeighborsRegressor
# 'gp'->GaussianProcessRegressor
# 'sgd'-> SGDRegressor
# 'voting'->VotingRegressor
# 'nnet'->MLPRegressor
# 'stacking'->StackingRegressor



y_pred = models['stacking'].predict(X_test_new)
#y_pred = models['rf'].predict(X_test_new)
#Compare predictions with ground truth
score = mean_absolute_error(y_test_new, y_pred)
print("MAE: ", score)

y_std = std(score)

reg_error = y_test_new-y_pred

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
#show the std?
ax.fill_between(n, y_pred-y_std, y_pred+y_std,
                color="pink", alpha=0.5, label="std")

ax.legend()
ax.grid(True)
pyplot.show()