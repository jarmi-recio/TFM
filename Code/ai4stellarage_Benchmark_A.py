import numpy as np
from numpy import mean
from sklearn import preprocessing
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import StackingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF, ExpSineSquared # define the kernels to use in the GPs
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV #To perform the search for the best parameters
from sklearn.metrics import mean_absolute_error #define the metrics to use for the evaluation
import matplotlib.pyplot as plt
from joblib import dump, load
import pandas as pd
plt.rcParams.update({'font.size': 16})

# the script imports the data from a data file with the information of the stars
def get_dataset():
	data = pd.read_csv('data/gyro_tot_v20180801.txt', sep="\t", header=0)
	df = data[['M', 'R', 'Teff','L','Meta','logg','Prot','Age','eAge1','eAge2','class']]
	# age limits, only for graphics
	df['low_age'] = df.Age - df.eAge1
	df['high_age'] = df.Age + df.eAge2
	# clean NA values
	df.dropna(inplace=True, axis=0)
	#Select Andy's favourites
	df = df.loc[(df['class'] == 'MS') & (df['M'] < 2) & (df['M'] > 0.7) & (df['Prot'] < 50)]
	#sort the dataframe by age
	df = df.sort_values(by=['Age'])
	#chose target variable: age
	y = np.array(df['Age'])
	#selection of the data to be used
	X = np.array(df[['M', 'R', 'Teff','L','Meta','logg','Prot']])
	return X, y, df

# get a stacking ensemble of models using the best parameters
def get_best_stacking():
	# define the base models
	level0 = list()
	level0.append(('nnet', MLPRegressor(activation='relu', hidden_layer_sizes=(50,50,50), learning_rate='adaptive',
										learning_rate_init=0.1,max_iter=2000, solver='sgd', alpha=0.01, random_state=0, verbose = False)))
	level0.append(('gp', GaussianProcessRegressor(kernel = WhiteKernel(noise_level=1) + RBF(length_scale=1) + DotProduct(sigma_0=1),
												  random_state = 0)))

	# define meta learner model
	level1 = LinearRegression()
	# define the stacking ensemble
	model = StackingRegressor(estimators=level0, final_estimator=level1, cv=2)
	return model

# get a list of models to evaluate
def get_models():
	models = dict()

	# Neural Networks
	models['nnet'] = MLPRegressor(activation='relu', hidden_layer_sizes=(50, 50, 50), learning_rate='adaptive',
								  learning_rate_init=0.1, max_iter=2000, solver='sgd', alpha=0.01, random_state=0,
								  verbose=True)

	# Linear Regression
	tuned_parameters_lr = [{'normalize': ['True','False']}]
	clf_lr = GridSearchCV(LinearRegression(), tuned_parameters_lr, scoring='neg_mean_absolute_error')
	models['lr'] = clf_lr

	# Decision Tree
	tuned_parameters_dtr = [{'min_samples_leaf': [5, 10, 50, 100],
							 'criterion' : ['mse', 'friedman_mse', 'mae', 'poisson'], 'splitter' : ['best', 'random'],
							 'random_state':[0]}]
	clf_dtr = GridSearchCV(DecisionTreeRegressor(), tuned_parameters_dtr, scoring='neg_mean_absolute_error')
	models['dtr'] = clf_dtr

	# Random Forest
	tuned_parameters_rf = [{'min_samples_leaf': [5, 10, 50, 100], 'n_estimators': [5, 10, 50, 100],
							 'criterion': ['mse', 'mae'], 'random_state':[0]}]
	clf_rf = GridSearchCV(RandomForestRegressor(), tuned_parameters_rf, scoring='neg_mean_absolute_error')
	models['rf'] = clf_rf

	# SVR
	tuned_parameters_svm = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]}]
	clf_svm = GridSearchCV(SVR(), tuned_parameters_svm, scoring='neg_mean_absolute_error')
	models['svm'] = clf_svm

	# Bayesian Ridge
	tuned_parameters_bayes = [{'n_iter': [100, 200, 300, 400, 500]}]
	clf_bayes = GridSearchCV(BayesianRidge(), tuned_parameters_bayes, scoring='neg_mean_absolute_error')
	models['bayes'] = clf_bayes

	# kNNeighbours
	tuned_parameters_knn = [{'n_neighbors': [1, 5, 10, 15, 20, 50], 'weights' : ['uniform','distance'],
							 'algorithm' : ['auto','ball_tree','kd_tree','brute']}]
	clf_knn = GridSearchCV(KNeighborsRegressor(), tuned_parameters_knn, scoring='neg_mean_absolute_error')
	models['knn'] = clf_knn

	# Gaussian Process
	tuned_parameters_gp = [{'kernel': [WhiteKernel() + RBF() + DotProduct(), RBF() + DotProduct()], 'random_state':[0]}]
	clf_gp = GridSearchCV(GaussianProcessRegressor(), tuned_parameters_gp, scoring='neg_mean_absolute_error')
	models['gp'] = clf_gp

	#Stacking
	models['stacking'] = get_best_stacking()

	return models

# evaluate a given model using a train/test split
def evaluate_model(model, X_train_norm, y_train, X_test_norm, y_test):
	model.fit(X_train_norm, y_train) #perform training
	y_pred_train = model.predict(X_train_norm)
	y_pred_test = model.predict(X_test_norm)
	score_train = mean_absolute_error(y_train, y_pred_train)
	score_test = mean_absolute_error(y_test, y_pred_test)

	df_evaluate_aux = pd.DataFrame(list(zip(y_test, y_pred_test)), columns=['y_test', 'y_pred'])
	df_evaluate_aux.to_csv('df_data_' + name + '.csv')

	return score_train, score_test

# define dataset
X, y, df = get_dataset()

# perform train test split (test 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# save train and data test as unique dataframe
df_data_aux1 = pd.DataFrame(list(zip(X_train, y_train)), columns =['X_train', 'y_train'])
df_data_aux2 = pd.DataFrame(list(zip(X_test, y_test)), columns =['X_test', 'y_test'])
df_data_1 = pd.concat([df_data_aux1, df_data_aux2], ignore_index=True, axis=1)
df_data_1.columns = ['X_train', 'y_train', 'X_test', 'y_test']

df_data_1.to_csv('df_data_B_A.csv')

# data normalization
scaler = preprocessing.StandardScaler().fit(X_train)
X_train_norm = scaler.transform(X_train)
X_test_norm = scaler.transform(X_test)

# get the models to evaluate
models = get_models()

# evaluate the models and store results
results_train, results_test, names = list(), list(), list()
for name, model in models.items():
	scores_train, scores_test = evaluate_model(model, X_train_norm, y_train, X_test_norm, y_test)
	results_train.append(scores_train)
	results_test.append(scores_test)
	names.append(name)
	print('>%s %.3f %.3f' % (name, mean(scores_train), mean(scores_test)))

df_results = pd.DataFrame(list(zip(names, results_train, results_test)), columns =['Name', 'MAE_train', 'MAE_test'])
print(df_results)
df_results.to_csv('df_results_B_A.csv')


## Bar Plot
fig, ax = plt.subplots(figsize=(10, 5))

ax.bar(names, results_test, 0.5)

for i, v in enumerate(results_test):
	ax.text(i - 0.35, v + 0.02, round(results_test[i], 4), fontsize=14)

# Remove the axes
for s in ['top', 'bottom', 'left', 'right']:
	ax.spines[s].set_visible(False)

# Add x, y gridlines
ax.grid(b=True, color='grey', linestyle='-.', linewidth=0.5, alpha=0.2)
ax.set_ylabel('M.A.E (Gyr)')
ax.set_xlabel('Models')
plt.show()

## Training histogram
num_bins = 15
fig, ax1 = plt.subplots(figsize=(10,5))

n1, bins1, patches1 = ax1.hist(y_train, num_bins, density=False, color='blue')

ax1.set_xlabel('Training ages (Gyr)')
ax1.set_ylabel('Number of Stars')
plt.show()

## Test histogram
fig, ax2 = plt.subplots(figsize=(10,5))
n2, bins2, patches2 = ax2.hist(y_test, num_bins, density=False, color='blue')

ax2.set_xlabel('Test ages (Gyr)')
ax2.set_ylabel('Number of Stars')
plt.show()

# process to add the age limits from no normalized data
X_test_df = pd.DataFrame(X_test, columns=['M', 'R', 'Teff','L','Meta','logg','Prot'])
y_test_df = pd.DataFrame(y_test)
X_test_df['Age'] = y_test_df

aux_df_2 = pd.merge(X_test_df, df, how="left", on=['M', 'R', 'Teff','L','Meta','logg','Prot'])
X_test_limits_df = aux_df_2[['M', 'R', 'Teff','L','Meta','logg','Prot','Age_y','low_age','high_age']]
X_test_limits_df = X_test_limits_df.sort_values(by='Age_y')

# rescue the normalized data
aux = pd.DataFrame(X_test_norm)
aux['Age'] = y_test
aux = aux.sort_values(by='Age')

X_test_new = aux.iloc[:,0:7]
y_test_new = aux['Age'].to_numpy()

perc = list()
for name, model in models.items():
	y_pred_model = models[name].predict(X_test_new)
	reg_error = y_test_new - y_pred_model

	df_final_test = pd.DataFrame(list(zip(y_test_new, y_pred_model, reg_error)), columns=['y_test', 'y_pred_model', 'reg_error'])
	df_final_test['low_age'] = X_test_limits_df['low_age'].values
	df_final_test['high_age'] = X_test_limits_df['high_age'].values

	df_model_B1_error_band = df_final_test.loc[((df_final_test['y_pred_model'] >= df_final_test['low_age']) &
											  (df_final_test['y_pred_model'] <= df_final_test['high_age']))]

	percent_B1 = round((len(df_model_B1_error_band)) * 100 / (len(df_final_test)), 3)
	perc.append(percent_B1)
	print('{}: {}% stars inside error band'.format(name,percent_B1))

	# figure 1
	fig, ax1 = plt.subplots(figsize=(10, 5))

	ax1.scatter(df_final_test.y_test, df_final_test.y_pred_model, color = 'blue')
	ax1.plot(range(0, 12), range(0, 12), color='black')
	ax1.fill_between(df_final_test.y_test, X_test_limits_df['low_age'], X_test_limits_df['high_age'], color="gray",
						   alpha=0.5, label="Margin")

	plt.yticks(np.arange(0, 15, 1))
	plt.xticks(np.arange(0, 13, 1))
	plt.xlabel("Age (Gyr)")
	plt.ylabel("Prediction (Gyr)")
	plt.legend(loc='upper left')
	plt.show()

	n = np.arange(df_final_test['y_pred_model'].size)

	#figure 2
	fig, ax2 = plt.subplots(figsize=(10, 5))
	ax2.plot(n, df_final_test['y_test'], c='tab:blue', label='Age')
	ax2.plot(n, df_final_test['y_pred_model'], c='tab:orange', label='Prediction')
	ax2.plot(n, abs(df_final_test['reg_error']), c='tab:red', label='Error')
	ax2.fill_between(n, X_test_limits_df['low_age'], X_test_limits_df['high_age'], color="gray", alpha=0.5, label="Margin")

	plt.xlabel('Cases')
	plt.ylabel('Age (Gyr)')
	plt.yticks(np.arange(0, 15, 1))
	plt.legend()
	plt.grid(True)
	plt.show()

	# histogram of mean errors by bin

	aux = []
	mean_error = []
	for i in range(len(bins2) - 1):
		if (i == 14):
			mean_err = (abs(df_final_test[(df_final_test['y_test'] >= bins2[i])
										& (df_final_test['y_test'] <= bins2[i + 1])].reg_error)).mean()
			mean_error.append(mean_err)
			aux_2 = (bins2[i] + bins2[i + 1]) / 2
			aux.append(aux_2)

		else:
			mean_err = (abs(df_final_test[(df_final_test['y_test'] >= bins2[i])
										& (df_final_test['y_test'] < bins2[i + 1])].reg_error)).mean()
			mean_error.append(mean_err)
			aux_2 = (bins2[i] + bins2[i + 1]) / 2
			aux.append(aux_2)

	fig, ax3 = plt.subplots(figsize=(10, 5))

	ax3.bar(aux, mean_error, color='orange')

	plt.xlabel('Age (Gyr)')
	plt.ylabel('Mean error (Gyr)')
	plt.show()

df_results_percent = pd.DataFrame(list(zip(names, perc)), columns =['Name', 'Percentage'])
print(df_results_percent)
df_results.to_csv('df_percent_B_A.csv')

dump(models,'results/models.joblib')