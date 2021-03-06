import numpy as np
from numpy import mean
from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from joblib import load, dump
import pandas as pd

# load model (and test data) for further analysis
models = load('results/models.joblib')
X_train, y_train, X_test_no, y_test_no = load('results/test_data_no_norm.joblib')


def get_dataset():
    data = pd.read_csv('data/test_gyro.txt', sep=",", header=0)
    # clean NA values
    df = data[['m', 'r', 'teff', 'L', 'Meta', 'logg', 'prot', 'ager']]
    df.dropna(inplace=True, axis=0)
    # sort the dataframe by age
    df = df.sort_values(by=['ager'])
    # chose target variable: age
    y = np.array(df['ager'])
    # selection of the data to be used
    X = np.array(df[['m', 'r', 'teff', 'L', 'Meta', 'logg', 'prot']])
    return X, y


# evaluate a given model using a train/test split
def evaluate_model(model, X_test, y_test):
    y_pred_test = model.predict(X_test)
    score_test = mean_absolute_error(y_test, y_pred_test)
    return score_test


# define dataset
X_test, y_test = get_dataset()

df_data_aux1 = pd.DataFrame(list(zip(X_train, y_train)), columns =['X_train', 'y_train'])
df_data_aux2 = pd.DataFrame(list(zip(X_test, y_test)), columns =['X_test', 'y_test'])
df_data = pd.concat([df_data_aux1, df_data_aux2], ignore_index=True, axis=1)
df_data.columns = ['X_train', 'y_train', 'X_test', 'y_test']

df_data.to_csv('df_data_new_test.csv')


# data normalization
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

results_test, names = list(), list()
for name, model in models.items():
    scores_test = evaluate_model(model, X_test, y_test)
    results_test.append(scores_test)
    names.append(name)
    print('>%s %.3f' % (name, mean(scores_test)))

df_results = pd.DataFrame(list(zip(names, results_test)), columns=['Name', 'MAE_test'])
print(df_results)

# plot model performance for comparison
plt.bar(names, results_test)
plt.show()
dump([X_train, y_train, X_test,y_test], 'results/test_data_for_new_test.joblib', compress=1)
df_results.to_csv('df_results_with_new_test.csv')
