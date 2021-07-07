""" Code for loading data. """
import numpy as np
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
            self.amp_range = config.get('amp_range', [0.1, 5.0])
            self.phase_range = config.get('phase_range', [0, np.pi])
            self.input_range = config.get('input_range', [-5.0, 5.0])
            self.dim_input = 1
            self.dim_output = 1
        else:
            raise ValueError('Unrecognized data source')


    def generate_sinusoid_batch(self, train=True, input_idx=None):

        amp = np.random.uniform(self.amp_range[0], self.amp_range[1], [self.batch_size])

        outputs = np.zeros([self.batch_size, self.num_samples_per_class, self.dim_output])
        init_inputs = np.zeros([self.batch_size, self.num_samples_per_class, self.dim_input])

        for func in range(self.batch_size):
            init_inputs[func] = np.random.uniform(self.input_range[0], self.input_range[1], [self.num_samples_per_class, 1])
            if input_idx is not None:
                init_inputs[:,input_idx:,0] = np.linspace(self.input_range[0], self.input_range[1], num=self.num_samples_per_class-input_idx, retstep=False)

            outputs[func] = amp[func] * np.sin(init_inputs[func])

        # figure
        fig, ax1 = plt.subplots(figsize=(10, 5))

        ax1.scatter(np.resize(init_inputs, (self.num_samples_per_class, 1)), np.resize(outputs, (self.num_samples_per_class, 1)),label="out")
        #ax1.scatter(range(0, self.num_samples_per_class), np.resize(init_inputs, (self.num_samples_per_class, 1)),label="in")

        plt.title("Sin")
        plt.xlabel("Time")
        plt.ylabel("Amp")
        plt.legend(loc='upper left')
        plt.show()

        return init_inputs, outputs

