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

# get the dataset the script imports the data from a data file with the information of the stars
def get_dataset():
	data = pd.read_csv('data/gyro_tot_v20180801.txt', sep="\t", header=0)
	#clean NA values (simply remove the corresponding columns)
	df = data[['Seq', 'M', 'R', 'Teff','L','Meta','logg','Prot','Age']]
	df.dropna(inplace=True, axis=0)
	#Chose target variable -> we play to estimate the age so
	y = np.array(df['Age'])
	#Selection of the data to be used
	X = np.array(df[['Seq', 'M', 'R', 'Teff','L','Meta','logg','Prot']])
	#normalize X
	X = preprocessing.scale(X)
	return X, y

def get_dataset_with_imputation():
	data = pd.read_csv('data/gyro_tot_v20180801.txt', sep="\t", header=0)
	#clean NA values (simply remove the corresponding columns)
	df = data[['Seq', 'M', 'R', 'Teff','L','Meta','logg','Prot','Age']]
	df.dropna(inplace=True, axis=0)
	print(df)
	#Chose target variable -> we play to estimate the age so
	y = np.array(df['Age'])
	#Selection of the data to be used
	X = np.array(df[['Seq', 'M', 'R', 'Teff','L','Meta','logg','Prot']])

	#perform data imputation with a regression model (not very good results)
	#imp = IterativeImputer(max_iter=10, random_state=0)  # configure data imputation
	#imp.fit(X)
	#X_test = X
	#X = imp.transform(X_test)

	#data imputation with the mean
	imp = KNNImputer()
	imp.fit(X)
	X = imp.transform(X)
	return X, y

# get a stacking ensemble of models
def get_stacking():
	# define the base models
	level0 = list()
	#level0.append(('lr',LinearRegression()))
	level0.append(('dtr', DecisionTreeRegressor()))
	level0.append(('rf', RandomForestRegressor(random_state=0)))
	level0.append(('svm', SVR()))
	#level0.append(('bayes', BayesianRidge()))
	level0.append(('knn', KNeighborsRegressor()))
	gp_kernel = WhiteKernel() + RBF() + DotProduct()
	level0.append(('gp', GaussianProcessRegressor(kernel=gp_kernel, random_state=0)))
	# define meta learner model
	#level1 = LinearRegression()
	#level1 = BayesianRidge()
	level1 = GaussianProcessRegressor(kernel=gp_kernel, random_state=0)
	# define the stacking ensemble
	model = StackingRegressor(estimators=level0, final_estimator=level1, cv=5)
	return model

# get a stacking ensemble of models using the best parameters
def get_best_stacking():
	# define the base models
	level0 = list()
	#level0.append(('lr',LinearRegression()))
	level0.append(('nnet', MLPRegressor(activation='tanh', hidden_layer_sizes=150,
										learning_rate='invscaling', max_iter=400, solver='lbfgs')))
	r1 = RandomForestRegressor(criterion='mse', min_samples_leaf=5, n_estimators=50, random_state=0)
	r2 = DecisionTreeRegressor(min_samples_leaf=5, criterion='friedman_mse', splitter='best')
	level0.append(('voting', VotingRegressor([('rf', r1), ('dtr', r2)])))
	level0.append(('dtr', DecisionTreeRegressor(min_samples_leaf = 5, criterion='friedman_mse', splitter='best')))
	level0.append(('rf', RandomForestRegressor(criterion='mse',min_samples_leaf=5, n_estimators=50, random_state=0)))
	#level0.append(('svm', SVR(kernel='rbf', C=1000, gamma=1e-3)))
	#level0.append(('bayes', BayesianRidge()))
	level0.append(('knn', KNeighborsRegressor(algorithm='auto',n_neighbors=5,weights='distance')))
	gp_kernel = WhiteKernel() + RBF() + DotProduct()
	level0.append(('gp', GaussianProcessRegressor(kernel=gp_kernel, random_state=0)))
	# define meta learner model
	#level1 = LinearRegression()
	#level1 = BayesianRidge()
	#level1 = GaussianProcessRegressor(kernel=gp_kernel, random_state=0)
	level1 = RandomForestRegressor(criterion='mse',min_samples_leaf=5, n_estimators=50, random_state=0)
	#level1 = DecisionTreeRegressor(min_samples_leaf = 5, criterion='friedman_mse', splitter='best')
	# define the stacking ensemble
	model = StackingRegressor(estimators=level0, final_estimator=level1, cv=5)
	return model

# get a list of models to evaluate
def get_models():
	models = dict()

	# Neural Networks
	tuned_parameters_nnet = [{'hidden_layer_sizes': [100, 150],
							  'max_iter': [200, 300, 400],
							  'activation': ['identity', 'logistic', 'tanh', 'relu'],
							  'solver': ['lbfgs', 'sgd', 'adam'],
							  'learning_rate': ['constant', 'invscaling', 'adaptive']}]
	clf_nnet = GridSearchCV(MLPRegressor(), tuned_parameters_nnet, scoring='neg_mean_absolute_error')
	models['nnet'] = clf_nnet

	#LinearRegression
	tuned_parameters_lr = [{'normalize': ['True','False']}]
	clf_lr = GridSearchCV(LinearRegression(), tuned_parameters_lr, scoring='neg_mean_absolute_error')
	models['lr'] = clf_lr

	#DecisionTree
	tuned_parameters_dtr = [{'min_samples_leaf': [5, 10, 50, 100],
							 'criterion' : ['mse', 'friedman_mse', 'mae', 'poisson'], 'splitter' : ['best', 'random']}]
	clf_dtr = GridSearchCV(DecisionTreeRegressor(), tuned_parameters_dtr, scoring='neg_mean_absolute_error')
	models['dtr'] = clf_dtr

	#RandomForest
	tuned_parameters_rf = [{'min_samples_leaf': [5, 10, 50, 100], 'n_estimators': [5, 10, 50, 100],
							 'criterion': ['mse', 'mae'] }]
	clf_rf = GridSearchCV(RandomForestRegressor(), tuned_parameters_rf, scoring='neg_mean_absolute_error')
	models['rf'] = clf_rf

	#SVR
	tuned_parameters_svm = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]}]
	clf_svm = GridSearchCV(SVR(), tuned_parameters_svm, scoring='neg_mean_absolute_error')
	models['svm'] = clf_svm

	#Bayesian Ridge
	tuned_parameters_bayes = [{'n_iter': [100, 200, 300, 400, 500]}]
	clf_bayes = GridSearchCV(BayesianRidge(), tuned_parameters_bayes, scoring='neg_mean_absolute_error')
	models['bayes'] = clf_bayes

	#kNeighbours
	tuned_parameters_knn = [{'n_neighbors': [1, 5, 10, 15, 20, 50], 'weights' : ['uniform','distance'],
							 'algorithm' : ['auto','ball_tree','kd_tree','brute']}]
	clf_knn = GridSearchCV(KNeighborsRegressor(), tuned_parameters_knn, scoring='neg_mean_absolute_error')
	models['knn'] = clf_knn

	#Gaussian Process
	#gp_kernel_1 = WhiteKernel() + RBF() + DotProduct()
	#gp_kernel_2 = ExpSineSquared() + RBF() + DotProduct()
	tuned_parameters_gp = [{'kernel': [WhiteKernel() + RBF() + DotProduct(), RBF() + DotProduct()],
							'random_state':[0,1]}]
	clf_gp = GridSearchCV(GaussianProcessRegressor(), tuned_parameters_gp, scoring='neg_mean_absolute_error')
	models['gp'] = clf_gp
	#models['gp'] = GaussianProcessRegressor(kernel=gp_kernel, random_state=0)

	#Stochastic Gradient Descent
	tuned_parameters_sgd = [{'loss': ['squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive']}]
	clf_sgd = GridSearchCV(SGDRegressor(), tuned_parameters_sgd, scoring='neg_mean_absolute_error')
	models['sgd'] = clf_sgd

	#Voting Regressor
	r1 = RandomForestRegressor(criterion='mse',min_samples_leaf=5, n_estimators=50, random_state=0)
	r2 = DecisionTreeRegressor(min_samples_leaf = 5, criterion='friedman_mse', splitter='best')
	models['voting'] = VotingRegressor([('rf', r1), ('dtr', r2)])

	#Option 1 -- Automatic search of parameters
	#tuned_params_stacking ={'dtr__min_samples_leaf': [5, 10, 50, 100],
	#						'dtr__criterion' : ['mse', 'friedman_mse', 'mae', 'poisson'],
	#						'dtr__splitter': ['best', 'random'],
	#						'rf__min_samples_leaf': [5, 10, 50, 100], 'rf__n_estimators': [5, 10, 50, 100],
	#						'rf__criterion': ['mse', 'mae'],
	#						'knn__n_neighbors': [1, 5, 10, 15, 20, 50], 'knn__weights': ['uniform', 'distance'],
	#						'knn__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
	#						}

	#clf_stacking = get_stacking()
	#models['stacking'] = GridSearchCV(clf_stacking, tuned_params_stacking, scoring='neg_mean_absolute_error')

	#Option 2 -- Use the best parameters ;) from a previous execution -->modify the parameters in the function get_best_stacking
	models['stacking'] = get_best_stacking()

	return models

# evaluate a given model using a train/test split
def evaluate_model(model, X_train, y_train, X_test, y_test):
	model.fit(X_train, y_train) #perform training
	y_pred = model.predict(X_test)
	score = mean_absolute_error(y_test, y_pred)
	#cv_scores = cross_val_score(model, X, y)
	#print(name)
	#print(model.best_params_)
	return score

# define dataset
#X, y = get_dataset_with_imputation()
X, y = get_dataset()
# perform train test split (test 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

#Normalize data
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# get the models to evaluate
models = get_models()
# evaluate the models and store results
results, names = list(), list()
for name, model in models.items():
	scores = evaluate_model(model, X_train, y_train, X_test, y_test)
	results.append(scores)
	names.append(name)
	print('>%s %.3f' % (name, mean(scores)))

df_results = pd.DataFrame(list(zip(names, results)), columns =['Name', 'Score'])
print(df_results)

# plot model performance for comparison
plt.bar(names, results)
plt.show()

# save model (and test data) for further analysis
dump(models,'results/models.joblib')
dump([X_test,y_test], 'results/test_data.joblib', compress=1)
df_results.to_csv('df_results.csv')
