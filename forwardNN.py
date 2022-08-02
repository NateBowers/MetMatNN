# Written and tested in Python 3.10.4
# See requirements.txt for the versions of packages needed

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import json
import shutil
import argparse
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner as kt

RANDOM_SEED = 42
METRICS = [
    'MeanSquaredError',
    'MeanAbsolutePercentageError',
]

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

class TrainingData:
    def __init__(self, path_x, path_y, percent_test=0.2):
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

        train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=float(percent_test),random_state=RANDOM_SEED)

        self.train_x = self.convert(train_x)
        self.train_y = self.convert(train_y)
        self.test_x = self.convert(test_x)
        self.test_y = self.convert(test_y)

    def convert(self, arr):
        tensor = tf.convert_to_tensor(arr, dtype=tf.float32)
        return tensor

    def save_stats(self, path):
        stats = {'mean':float(self.mean), 'std':float(self.std)}
        save_json(stats, 'stats', path)

    def IO_shape(self):
        input_shape = self.train_x.shape[1]
        output_shape = self.train_y.shape[1]
        return (input_shape, output_shape)


class ModelTuner(kt.HyperModel):
    def __init__ (self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def build(self, hp):
        model = keras.Sequential()
        model.add(layers.Input(shape=self.input_shape))
        for i in range(4):
            model.add(
                layers.Dense(
                    units=hp.Int(f"units_{i}", min_value=128, max_value=512, step=32),
                    activation=hp.Choice(f"activation_{i}", ["relu", "selu"]),
                )
            )
        model.add(layers.Dense(units=self.output_shape))
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=hp.Float("lr", min_value=0.0001, max_value=0.01, step=0.0001)),
            loss='mean_squared_error',
            metrics=METRICS,
        )
        return model

    def fit(self, hp, model, data, num_epochs, val_percent=0.2, **kwargs):
        return model.fit(
            x=data.train_x,
            y=data.train_y,
            batch_size=hp.Int("batch_size", min_value=32, max_value=128, step=16),
            epochs=num_epochs,
            verbose=0,
            validation_split=val_percent,
        )

def main(x_path, y_path, output_path, num_epochs):
    if os.path.exists(output_path):
        shutil.rmtree(output_path)

    data = TrainingData(x_path, y_path)
    data.save_stats(output_path)
    (input_shape, output_shape) = data.IO_shape()

    tuner = kt.BayesianOptimization(
        ModelTuner(input_shape=input_shape, output_shape=output_shape),
        objective=kt.Objective("val_mean_squared_error", direction="min"),        
        max_trials=100,
        executions_per_trial=1,
        directory=output_path,
        project_name='Tune_MetMatNN',
    )

    tuner.search( 
        data=data,
        num_epochs=num_epochs,
    )

    tuner.results_summary()

    hpmodel = ModelTuner(input_shape, output_shape)
    best_hp = tuner.get_best_hyperparameters()[0]
    print(type(tuner.results_summary()))
    model = hpmodel.build(best_hp)
    history = hpmodel.fit(best_hp, model, data, num_epochs=1, val_percent=0)

    print(model.summary)
    scores = model.evaluate(data.test_x, data.test_y, return_dict=True)

    model.save(os.path.join(output_path, 'model'))
    save_json(history.history, 'histories', output_path)
    save_json(scores, 'scores', output_path)
    for key in history.history:
        plot_vs_epoch(key, output_path, history.history[key])
        
    input_val = tf.reshape(data.test_x[0], shape=[1, data.test_x.shape[1]])
    predicted = model.predict(input_val).flatten()
    actual = data.test_y[0].numpy().flatten()
    plot_vs_epoch('Cross sectional scattering', output_path, predicted, actual, compare=True, display=True)


if __name__=="__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument("--x_path",type=str,default='data/8_layer_tio2_val.csv')
    parser.add_argument("--y_path",type=str,default='data/8_layer_tio2.csv')
    parser.add_argument("--output_path",type=str,default='results')
    parser.add_argument("--num_epochs",type=int,default='50')

    args = parser.parse_args()
    arg_dict = vars(args)

    kwargs = {  
        'x_path':arg_dict['x_path'],
        'y_path':arg_dict['y_path'],
        'output_path':arg_dict['output_path'],
        'num_epochs':arg_dict['num_epochs'],
    }

    main(**kwargs)