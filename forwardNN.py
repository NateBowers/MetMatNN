# Written and tested in Python 3.10.4
# See requirements.txt for the versions of packages needed

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import glorot_normal
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.integration.keras import TuneReportCallback
import os
import json
import argparse
import shutil

# Initialize Constants
# RANDOM_SEED is a random number used to make results replicable
# METRICS are the different error functions tracked
# CONFIG dict is the hyperparameters of the network
# HP_CONFIG dict is the hyperparameter space searched when tuning the network
RANDOM_SEED = 42
METRICS = [
    'MeanAbsoluteError', 
    'MeanAbsolutePercentageError'
]
CONFIG = {
    'lr': 0.001,
    'batch_size': 64,
    'num_epochs': 2,
    'num_hidden': 3,
    'num_nodes0': 256,
    'num_nodes1': 256,
    'num_nodes2': 256,
    'num_nodes3': 256,
    'dropout_rate0': 0.3,
    'dropout_rate1': 0.3,
    'dropout_rate2': 0.3,
    'dropout_rate3': 0.3,
}
HP_CONFIG = {
    'lr': tune.uniform(0.0001,0.1),
    'batch_size': tune.qrandint(16, 128, 16),
    'num_epochs': 150,
    'num_hidden': tune.randint(1,5), # the upper limit is exclusive, so options are (1,2,3,4)
    'num_nodes0': tune.qrandint(64, 320, 64),
    'num_nodes1': tune.qrandint(64, 320, 64),
    'num_nodes2': tune.qrandint(64, 320, 64),
    'num_nodes3': tune.qrandint(64, 320, 64),
    'dropout_rate0': tune.quniform(0.3, 0.8, 0.1),
    'dropout_rate1': tune.quniform(0.3, 0.8, 0.1),
    'dropout_rate2': tune.quniform(0.3, 0.8, 0.1),
    'dropout_rate3': tune.quniform(0.3, 0.8, 0.1),
}


class TrainingData:
    def __init__(self, path_x, path_y, percent_test=0.4):
        path_x = os.path.abspath(path_x)
        path_y = os.path.abspath(path_y)

        data_x = np.genfromtxt(path_x, delimiter=',', dtype='float32')
        data_y = np.genfromtxt(path_y, delimiter=',', dtype='float32')

        if data_x.shape[0] == data_y.shape[0]:
            pass
        elif data_x.shape[0] == data_y.shape[1] or data_x.shape[1] == data_y.shape[0]:
            data_y = np.transpose(data_y)
        else:
            raise Exception("x and y data arrays do not share a common dimension.")

        mean, std = data_x.mean(), data_x.std()
        data_x = (data_x-mean)/std

        self.mean = mean
        self.std = std

        self.data_x = data_x
        self.data_y = data_y

        train_x, test_x, train_y, test_y = train_test_split(self.data_x, self.data_y, test_size=float(percent_test),random_state=RANDOM_SEED)
        val_x, test_x, val_y, test_y = train_test_split(test_x, test_y, test_size=0.5,random_state=RANDOM_SEED)

        self.train_x = self.convert(train_x)
        self.train_y = self.convert(train_y)
        self.val_x = self.convert(val_x)
        self.val_y = self.convert(val_y)
        self.test_x = self.convert(test_x)
        self.test_y = self.convert(test_y)

    def convert(self, arr):
        tensor = tf.convert_to_tensor(arr, dtype=tf.float32)
        return tensor

    def save_stats(self, path):
        stats = {'mean':float(self.mean), 'std':float(self.std)}
        save_json(stats, 'stats', path)

def plot_vs_epoch(name, output_path, values1, values2=None, compare=False, display=False):
    plt.close()
    path = str(output_path + '/figures')
    save_path = os.path.join(path, name)
    if not os.path.exists(path):
        os.mkdir(path)
    if not compare:
        plt.plot(values1)
        plt.xlabel("Epoch")
        plt.title(name + "vs. Epoch")
    elif compare:
        plt.plot(values1, label='Predicted')
        plt.plot(values2, label='Actual')
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Cross Scatting Amplitude")
        plt.legend()
    plt.title(name)
    plt.savefig(save_path)
    if display:
        plt.show()

def save_json(data, name, output_path):
    filename = str(output_path + '/' + name + '.json')
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

def train_model(config, data, tuning=False):
    input_shape, output_shape = data.train_x.shape, data.train_y.shape
    initializer = glorot_normal(RANDOM_SEED)
    if tuning:
        callbacks = [TuneReportCallback({"mean_squared_error": "mean_squared_error"})]
    else:
        callbacks = []

    model = Sequential()
    model.add(Input(shape=(input_shape[1])))
    for i in range(int(config['num_hidden'])):
        nodes_call = 'num_nodes' + str(i)
        dropout_call = 'dropout_rate' + str(i)
        model.add(Dense(
            config[nodes_call], 
            activation='relu', 
            kernel_initializer=initializer,
        ))
        model.add(Dropout(config[dropout_call]))
    model.add(Dense(output_shape[1]))
    model.compile(
        optimizer=Adam(learning_rate=config['lr']), 
        loss='mean_squared_error', 
        metrics=METRICS,
    )

    history = model.fit(
        data.train_x, 
        data.train_y, 
        epochs=config['num_epochs'],
        batch_size=config['batch_size'], 
        validation_data=(data.val_x, data.val_y), 
        verbose=2,
        callbacks=callbacks,
    )

    if not tuning:
        return model, history

def tune_model(config, data):
    # sched = AsyncHyperBandScheduler(
    #     time_attr="training_iteration", max_t=400, grace_period=20
    # )

    analysis = tune.run(
        train_model(data, tuning=True),
        name="exp",
        scheduler="AsyncHyperBand",
        metric="mean_squared_error",
        mode="min",
        stop={"mean_squared_error": 0.01},
        num_samples=100,
        # resources_per_trial={"cpu": 2, "gpu": 0},
        config=config
    )
    print("Best hyperparameters found were: ", analysis.best_config)
    return analysis


def main(config, x_path, y_path, output_path):
    # Initialize data
    # Tune (ideally use parallel computing to speed up)
    # Save performance info (but NOT graphs and stuff
    # Use best config to run final thing
    # Save data
    

    if os.path.exists(output_path):
        shutil.rmtree(output_path)

    data = TrainingData(x_path, y_path)
    data.save_stats(output_path)

    model, history = train_model(config, data)
    scores = model.evaluate(data.test_x, data.test_y, return_dict=True)

    model.save(os.path.join(output_path, 'model'))
    save_json(history.history, 'histories', output_path)
    save_json(scores, 'scores', output_path)
    for key in history.history:
        plot_vs_epoch(key, output_path, history.history[key])
        
    # Plot example prediction
    input_val = tf.reshape(data.test_x[0], shape=[1, data.test_x.shape[1]])
    predicted = model.predict(input_val).flatten()
    actual = data.test_y[0].numpy().flatten()
    plot_vs_epoch('Cross sectional scattering', output_path, predicted, actual, compare=True, display=True)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--x_path",type=str,default='data/8_layer_tio2_val.csv')
    parser.add_argument("--y_path",type=str,default='data/8_layer_tio2.csv')
    parser.add_argument("--output_path",type=str,default='results/test1')

    args = parser.parse_args()
    dict = vars(args)

    kwargs = {  
        'x_path':dict['x_path'],
        'y_path':dict['y_path'],
        'output_path':dict['output_path'],
    }

    main(config=CONFIG, **kwargs)