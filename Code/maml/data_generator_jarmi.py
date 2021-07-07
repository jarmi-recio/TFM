""" Code for loading data. """
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from numpy.random import default_rng


import matplotlib.pyplot as plt


from tensorflow.python.platform import flags

FLAGS = flags.FLAGS

class DataGenerator(object):
    """
    Data Generator capable of generating batches of sinusoid or Omniglot data.
    A "class" is considered a class of omniglot digits or a particular sinusoid function.
    """
    def __init__(self, num_samples_per_class, batch_size, config={}):
        """
        Args:
            num_samples_per_class: num samples to generate per class in one batch
            batch_size: size of meta batch size (e.g. number of functions)
        """
        self.batch_size = batch_size
        self.num_samples_per_class = num_samples_per_class
        self.num_classes = 1  # by default 1 (only relevant for classification problems)

        if FLAGS.datasource == 'sinusoid':
            self.generate = self.generate_sinusoid_batch
            self.batch_size = batch_size
            self.amp_range = config.get('amp_range', [0.1, 5.0])
            self.phase_range = config.get('phase_range', [0, np.pi])
            self.input_range = config.get('input_range', [-5.0, 5.0])
            self.dim_input = 1
            self.dim_output = 1
        elif FLAGS.datasource == 'stars':
            self.batch_size = batch_size
            self.load = self.load_stars
            self.generate = self.generate_stars_batch
            self.dim_input = 1
            self.dim_output = 1
        else:
            raise ValueError('Unrecognized data source')


    def generate_sinusoid_batch(self, train=True, input_idx=None):
        # Note train arg is not used (but it is used for omniglot method.
        # input_idx is used during qualitative testing --the number of examples used for the grad update
        amp = np.random.uniform(self.amp_range[0], self.amp_range[1], [self.batch_size])
        phase = np.random.uniform(self.phase_range[0], self.phase_range[1], [self.batch_size])
        outputs = np.zeros([self.batch_size, self.num_samples_per_class, self.dim_output])
        init_inputs = np.zeros([self.batch_size, self.num_samples_per_class, self.dim_input])
        for func in range(self.batch_size):
            init_inputs[func] = np.random.uniform(self.input_range[0], self.input_range[1], [self.num_samples_per_class, 1])
            if input_idx is not None:
                init_inputs[:,input_idx:,0] = np.linspace(self.input_range[0], self.input_range[1], num=self.num_samples_per_class-input_idx, retstep=False)
            outputs[func] = amp[func] * np.sin(init_inputs[func]-phase[func])
        # figure
        '''fig, ax1 = plt.subplots(figsize=(10, 5))

        ax1.scatter(np.resize(init_inputs, (self.num_samples_per_class, 1)), np.resize(outputs, (self.num_samples_per_class, 1)),label="out")

        plt.title("Sin")
        plt.xlabel("Time")
        plt.ylabel("Amp")
        plt.legend(loc='upper left')
        plt.show()'''

        return init_inputs, outputs, amp, phase


    def load_stars(self, train=True, input_idx=None):

        def get_dataset():
            df_source = pd.read_csv('datasets/gyro_tot_v20180801.txt', sep="\t", header=0)
            # Required columns
            df = df_source.loc[:,
                 ['M', 'R', 'Teff', 'L', 'Meta', 'logg', 'Prot', 'Age', 'eM1', 'eM2', 'eR1', 'eR2', 'eTeff1', 'eTeff2',
                  'eL1', 'eL2', 'eMeta1', 'eMeta2', 'elogg1', 'elogg2', 'eProt1', 'eProt2', 'eAge1', 'eAge2', 'class']]
            # Filter due to gyro-phy
            df = df.loc[(df['class'] == 'MS') & (df['M'] < 2) & (df['M'] > 0.7) & (df['Prot'] < 50)]
            # Clean NA values
            df.dropna(inplace=True, axis=0)
            df.sort_values(by=['Age'])

            # Selection of the data to be used
            X_ser = df.loc[:, ['Teff']]
            X = X_ser.to_numpy()
            y_ser = df.loc[:, ['Age']]
            y = y_ser.to_numpy()

            # Import errors
            X_errors_ser = df.loc[:,['eTeff1', 'eTeff2']]
            X_errors = X_errors_ser.to_numpy()
            y_errors_ser = df.loc[:, ['eAge1', 'eAge2']]
            y_errors = y_errors_ser.to_numpy()

            # Appply uncertainties to generate data
            n_samples = 20  # number of samples to generate within the bounds provided in the dataset

            # Generate samples for every input point using a uniform distribution
            # Initialize random number generator
            seed = 1
            rng = default_rng(seed)

            # Initialize the output
            X_aug = X
            y_aug = y

            # iterate over the arrays simultaneosly
            for (s_x, s_xe, s_y, s_ye) in zip(X, X_errors, y, y_errors):
                y_new = rng.uniform(s_y - s_ye[0], s_y + s_ye[1], n_samples).reshape(n_samples, 1)
                y_aug = np.concatenate((y_aug, y_new), axis=0)

                X_new = rng.uniform(s_x[0] - s_xe[0], s_x[0] + s_xe[1], n_samples).reshape(n_samples, 1)
                X_aug = np.concatenate((X_aug, X_new), axis=0)

            X_final = X_aug
            y_final = y_aug

            return X_final, y_final, df


        def get_dataset_2():
            df_source = pd.read_csv('datasets/gyro_tot_v20180801.txt', sep="\t", header=0)
            # Required columns
            df = df_source.loc[:,
                 ['M', 'R', 'Teff', 'L', 'Meta', 'logg', 'Prot', 'Age', 'eM1', 'eM2', 'eR1', 'eR2', 'eTeff1', 'eTeff2',
                  'eL1', 'eL2', 'eMeta1', 'eMeta2', 'elogg1', 'elogg2', 'eProt1', 'eProt2', 'eAge1', 'eAge2','class']]
            # Filter due to gyro-phy
            df = df.loc[(df['class'] == 'MS') & (df['M'] < 2) & (df['M'] > 0.7) & (df['Prot'] < 50)]
            # Clean NA values
            df.dropna(inplace=True, axis=0)
            df.sort_values(by=['Age'])


            # Selection of the data to be used
            X_ser = df.loc[:, ['M', 'R', 'Teff', 'L', 'Meta', 'logg', 'Prot']]
            X = X_ser.to_numpy()
            y_ser = df.loc[:, ['Age']]
            y = y_ser.to_numpy()

            # Import errors
            X_errors_ser = df.loc[:,
                           ['eM1', 'eM2', 'eR1', 'eR2', 'eTeff1', 'eTeff2', 'eL1', 'eL2', 'eMeta1', 'eMeta2', 'elogg1',
                            'elogg2', 'eProt1', 'eProt2']]
            X_errors = X_errors_ser.to_numpy()
            y_errors_ser = df.loc[:, ['eAge1', 'eAge2']]
            y_errors = y_errors_ser.to_numpy()

            # Appply uncertainties to generate data
            n_samples = 20  # number of samples to generate within the bounds provided in the dataset

            # Generate samples for every input point using a uniform distribution
            # Initialize random number generator
            seed = 1
            rng = default_rng(seed)

            # Initialize the output
            X_aug = X
            y_aug = y

            # iterate over the arrays simultaneosly
            for (s_x, s_xe, s_y, s_ye) in zip(X, X_errors, y, y_errors):
                y_new = rng.uniform(s_y - s_ye[0], s_y + s_ye[1], n_samples).reshape(n_samples, 1)
                y_aug = np.concatenate((y_aug, y_new), axis=0)

                X_new = np.concatenate((rng.uniform(s_x[0] - s_xe[0], s_x[0] + s_xe[1], (n_samples, 1)),
                                        rng.uniform(s_x[1] - s_xe[2], s_x[1] + s_xe[3], (n_samples, 1)),
                                        rng.uniform(s_x[2] - s_xe[4], s_x[2] + s_xe[5], (n_samples, 1)),
                                        rng.uniform(s_x[3] - s_xe[6], s_x[3] + s_xe[7], (n_samples, 1)),
                                        rng.uniform(s_x[4] - s_xe[8], s_x[4] + s_xe[9], (n_samples, 1)),
                                        rng.uniform(s_x[5] - s_xe[10], s_x[5] + s_xe[11], (n_samples, 1)),
                                        rng.uniform(s_x[6] - s_xe[12], s_x[6] + s_xe[13], (n_samples, 1))), axis=1)

                X_aug = np.concatenate((X_aug, X_new), axis=0)

            X_final = X_aug
            y_final = y_aug
            df_final = np.append(X_final, y_final, axis=1)

            df_final = df_final[np.argsort(df_final[:, 7])]
            X_sorted, y_sorted = df_final[:, 0:7], df_final[:, 7]
            y_sorted = np.resize(y_sorted, (13251, 1))

            return X_sorted, y_sorted, df_final

        # define dataset
        X, y, df = get_dataset_2()

        # perform train test split (test 20%)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle = False)

        # data normalization
        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train_norm = scaler.transform(X_train)
        X_test_norm = scaler.transform(X_test)


        return X_train_norm, X_test_norm, y_train, y_test



    def generate_stars_batch(self, X_train, X_test, y_train, y_test, itr, train=True, input_idx=None):


        X_train_batch = np.zeros([self.batch_size, self.dim_output, len(X_train[0])])
        y_train_batch = np.zeros([self.batch_size, self.dim_input, len(y_train[0])])
        X_test_batch = np.zeros([self.batch_size, self.dim_output, len(X_test[0])])
        y_test_batch = np.zeros([self.batch_size, self.dim_input, len(y_test[0])])


        for ind, cont in enumerate(range(itr*self.batch_size, itr*self.batch_size + self.batch_size)):
            if FLAGS.train == True:
                X_train_batch[ind] = X_train[[cont],:]
                y_train_batch[ind] = y_train[[cont],:]

            elif FLAGS.train == False:
                X_test_batch[ind] = X_test[[cont],:]
                y_test_batch[ind] = y_test[[cont],:]


        return X_train_batch, X_test_batch, y_train_batch, y_test_batch
