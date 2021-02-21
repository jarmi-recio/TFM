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
	level0.append(('nnet', MLPRegressor(activation='logistic', hidden_layer_sizes=150, learning_rate='adaptive',
										max_iter=400, solver='lbfgs')))
	level0.append(('dtr', DecisionTreeRegressor(min_samples_leaf=5, criterion='mae', splitter='random')))
	level0.append(('rf', RandomForestRegressor(criterion='mae', min_samples_leaf=5, n_estimators=5, random_state=0)))
	level0.append(('knn', KNeighborsRegressor(algorithm='auto',n_neighbors=1,weights='uniform')))

	# define meta learner model
	level1 = DecisionTreeRegressor(min_samples_leaf = 5, criterion='mae', splitter='random')

	# define the stacking ensemble
	model = StackingRegressor(estimators=level0, final_estimator=level1, cv=5)
	return model

# get a list of models to evaluate
def get_models():
	models = dict()

	# Neural Networks
	# best params: {'activation': 'logistic', 'hidden_layer_sizes': 150, 'learning_rate': 'adaptive', 'max_iter': 400, 'solver': 'lbfgs'}
	tuned_parameters_nnet = [{'hidden_layer_sizes': [100, 150],
							  'max_iter': [200, 300, 400],
							  'activation': ['identity', 'logistic', 'tanh', 'relu'],
							  'solver': ['lbfgs', 'sgd', 'adam'],
							  'learning_rate': ['constant', 'invscaling', 'adaptive']}]
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

	#Stochastic Gradient Descent
	# best params: {'loss': 'epsilon_insensitive'}
	tuned_parameters_sgd = [{'loss': ['squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive']}]
	clf_sgd = GridSearchCV(SGDRegressor(), tuned_parameters_sgd, scoring='neg_mean_absolute_error')
	models['sgd'] = clf_sgd

	#Voting Regressor
	r1 = RandomForestRegressor(criterion='mse',min_samples_leaf=5, n_estimators=100, random_state=0)
	r2 = DecisionTreeRegressor(min_samples_leaf = 5, criterion='mae', splitter='best')
	models['voting'] = VotingRegressor([('rf', r1), ('dtr', r2)])

	#Use the best parameters from a previous execution --> modify the parameters in the function get_best_stacking
	models['stacking'] = get_best_stacking()

	return models

# evaluate a given model using a train/test split
def evaluate_model(model, X_train, y_train, X_test, y_test):
	model.fit(X_train, y_train) #perform training
	y_pred = model.predict(X_test)
	y_std = std(y_pred)
	score = mean_absolute_error(y_test, y_pred)
	#print(name)
	#print(model.best_params_)
	return score, y_std

# define dataset
X, y = get_dataset()

# perform train test split (test 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# perform train test split without randomization
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None, shuffle=False)

# switching train and test sets
'''aux_1 = X_train
aux_2 = y_train
X_train = X_test
y_train = y_test
X_test = aux_1
y_test = aux_2'''

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
results, names, y_stds = list(), list(), list()
for name, model in models.items():
	scores, y_std = evaluate_model(model, X_train, y_train, X_test, y_test)
	results.append(scores)
	y_stds.append(y_std)
	names.append(name)
	print('>%s %.3f' % (name, mean(scores)))

df_results = pd.DataFrame(list(zip(names, results, y_stds)), columns =['Name', 'MAE','std'])
print(df_results)

# plot model performance for comparison
plt.bar(names, results)
plt.show()

# save model (and test data) for further analysis
dump(models,'results/models.joblib')
dump([X_test,y_test], 'results/test_data.joblib', compress=1)
df_results.to_csv('df_results.csv')
