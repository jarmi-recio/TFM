import numpy as np
from numpy import mean
from numpy import std
from sklearn.datasets import make_regression
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import StackingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF, ExpSineSquared # define the kernels to use in the GPs
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV #To perform the search for the best parameters
from sklearn.metrics import mean_absolute_error #define the metrics to use for the evaluation
from matplotlib import pyplot
import matplotlib.pyplot as plt
from joblib import dump, load
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer

# the script imports the data from a data file with the information of the stars
def get_dataset():
	data = pd.read_csv('data/gyro_tot_v20180801.txt', sep="\t", header=0)
	#clean NA values
	df = data[['M', 'R', 'Teff','L','Meta','logg','Prot','Age']]
	df.dropna(inplace=True, axis=0)
	#sort the dataframe by age
	df = df.sort_values(by=['Age'])
	#chose target variable: age
	y = np.array(df['Age'])
	#selection of the data to be used
	X = np.array(df[['M', 'R', 'Teff','L','Meta','logg','Prot']])
	#normalize X
	X = preprocessing.scale(X)
	return X, y

# get a stacking ensemble of models using the best parameters
def get_best_stacking():
	# define the base models
	level0 = list()
	level0.append(('nnet', MLPRegressor(activation='tanh', hidden_layer_sizes=150, learning_rate='invscaling',
										max_iter=200, solver='lbfgs', alpha= 1e-05)))
	level0.append(('dtr', DecisionTreeRegressor(min_samples_leaf=5, criterion='mae', splitter='random')))
	level0.append(('rf', RandomForestRegressor(criterion='mae', min_samples_leaf=5, n_estimators=5, random_state=0)))
	level0.append(('knn', KNeighborsRegressor(algorithm='auto',n_neighbors=1,weights='uniform')))
	level0.append(('gp', GaussianProcessRegressor(kernel = WhiteKernel(noise_level=1) + RBF(length_scale=1) + DotProduct(sigma_0=1),
												  random_state = 0)))
	# define meta learner model
	level1 = GaussianProcessRegressor(kernel= WhiteKernel(noise_level=1) + RBF(length_scale=1) + DotProduct(sigma_0=1), random_state= 0)

	# define the stacking ensemble
	model = StackingRegressor(estimators=level0, final_estimator=level1, cv=5)
	return model

# get a list of models to evaluate
def get_models():
	models = dict()

	# Neural Networks
	# best params_2nd: {'activation': 'logistic', 'hidden_layer_sizes': 150, 'learning_rate': 'adaptive', 'max_iter': 400, 'solver': 'lbfgs'}
	# best params_1st: {'activation': 'tanh', 'alpha': 1e-05, 'hidden_layer_sizes': 150, 'learning_rate': 'invscaling', 'max_iter': 200,
	#'solver': 'lbfgs'}
	tuned_parameters_nnet = [{'hidden_layer_sizes': [100, 150],
							  'max_iter': [200, 300, 400],
							  'activation': ['identity', 'logistic', 'tanh', 'relu'],
							  'solver': ['lbfgs', 'sgd', 'adam'],
							  'learning_rate': ['constant', 'invscaling', 'adaptive'],
							  'alpha':[0.00001]}]
	clf_nnet = GridSearchCV(MLPRegressor(), tuned_parameters_nnet, scoring='neg_mean_absolute_error')
	models['nnet'] = clf_nnet

	#LinearRegression
	# best params: {'normalize': 'True'}
	tuned_parameters_lr = [{'normalize': ['True','False']}]
	clf_lr = GridSearchCV(LinearRegression(), tuned_parameters_lr, scoring='neg_mean_absolute_error')
	models['lr'] = clf_lr

	#DecisionTree
	# best params: {'criterion': 'friedman_mse', 'min_samples_leaf': 5, 'splitter': 'best'}
	tuned_parameters_dtr = [{'min_samples_leaf': [5, 10, 50, 100],
							 'criterion' : ['mse', 'friedman_mse', 'mae', 'poisson'], 'splitter' : ['best', 'random']}]
	clf_dtr = GridSearchCV(DecisionTreeRegressor(), tuned_parameters_dtr, scoring='neg_mean_absolute_error')
	models['dtr'] = clf_dtr

	#RandomForest
	# best params: {'criterion': 'mse', 'min_samples_leaf': 5, 'n_estimators': 100}
	tuned_parameters_rf = [{'min_samples_leaf': [5, 10, 50, 100], 'n_estimators': [5, 10, 50, 100],
							 'criterion': ['mse', 'mae'] }]
	clf_rf = GridSearchCV(RandomForestRegressor(), tuned_parameters_rf, scoring='neg_mean_absolute_error')
	models['rf'] = clf_rf

	#SVR
	# best params: {'C': 1000, 'gamma': 0.001, 'kernel': 'rbf'}
	tuned_parameters_svm = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]}]
	clf_svm = GridSearchCV(SVR(), tuned_parameters_svm, scoring='neg_mean_absolute_error')
	models['svm'] = clf_svm

	#Bayesian Ridge
	# best params: {'n_iter': 100}
	tuned_parameters_bayes = [{'n_iter': [100, 200, 300, 400, 500]}]
	clf_bayes = GridSearchCV(BayesianRidge(), tuned_parameters_bayes, scoring='neg_mean_absolute_error')
	models['bayes'] = clf_bayes

	#kNeighbours
	# best params: {'algorithm': 'auto', 'n_neighbors': 5, 'weights': 'distance'}
	tuned_parameters_knn = [{'n_neighbors': [1, 5, 10, 15, 20, 50], 'weights' : ['uniform','distance'],
							 'algorithm' : ['auto','ball_tree','kd_tree','brute']}]
	clf_knn = GridSearchCV(KNeighborsRegressor(), tuned_parameters_knn, scoring='neg_mean_absolute_error')
	models['knn'] = clf_knn

	#Gaussian Process
	# best params: {'kernel': WhiteKernel(noise_level=1) + RBF(length_scale=1) + DotProduct(sigma_0=1), 'random_state': 0}
	tuned_parameters_gp = [{'kernel': [WhiteKernel() + RBF() + DotProduct(), RBF() + DotProduct()],
							'random_state':[0,1]}]
	clf_gp = GridSearchCV(GaussianProcessRegressor(), tuned_parameters_gp, scoring='neg_mean_absolute_error')
	models['gp'] = clf_gp

	#Stacking with GaussianProcessRegressor in layer 1
	models['stacking'] = get_best_stacking()

	return models

# evaluate a given model using a train/test split
def evaluate_model(model, X_train, y_train, X_test, y_test):
	model.fit(X_train, y_train) #perform training
	y_pred_train = model.predict(X_train)
	y_pred_test = model.predict(X_test)
	score_train = mean_absolute_error(y_train, y_pred_train)
	score_test = mean_absolute_error(y_test, y_pred_test)
	#print(name)
	#print(model.best_params_)
	return score_train, score_test

# define dataset
X, y = get_dataset()

# perform train test split (test 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=None, shuffle=False)

# switching train and test sets
#aux_1 = X_train
#aux_2 = y_train
#X_train = X_test
#y_train = y_test
#X_test = aux_1
#y_test = aux_2

# save train and data test as unique dataframe
df_data_aux1 = pd.DataFrame(list(zip(X_train, y_train)), columns =['X_train', 'y_train'])
df_data_aux2 = pd.DataFrame(list(zip(X_test, y_test)), columns =['X_test', 'y_test'])
df_data = pd.concat([df_data_aux1, df_data_aux2], ignore_index=True, axis=1)
df_data.columns = ['X_train', 'y_train', 'X_test', 'y_test']

df_data.to_csv('df_data.csv')

# data normalization
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# get the models to evaluate
models = get_models()
# evaluate the models and store results
results_train, results_test, names = list(), list(), list()
for name, model in models.items():
	scores_train, scores_test = evaluate_model(model, X_train, y_train, X_test, y_test)
	results_train.append(scores_train)
	results_test.append(scores_test)
	names.append(name)
	print('>%s %.3f %.3f' % (name, mean(scores_train), mean(scores_test)))

df_results = pd.DataFrame(list(zip(names, results_train, results_test)), columns =['Name', 'MAE_train', 'MAE_test'])
print(df_results)

# plot model performance for comparison
plt.bar(names, results_test)
plt.show()

# save model (and test data) for further analysis
dump(models,'results/models.joblib')
dump([X_train, y_train, X_test,y_test], 'results/test_data.joblib', compress=1)
df_results.to_csv('df_results.csv')
