# Written and tested in Python 3.10.4

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.initializers import glorot_normal
from tensorboard.plugins.hparams import api as hp
from ray import tune
import os
import time
import json
import argparse
import shutil

# Initialize Constants
RANDOM_SEED = 42
METRICS = ['MeanSquaredError', 'RootMeanSquaredError', 'MeanAbsoluteError', 'MeanAbsolutePercentageError']

def load_data(data_path:str, percent_test=0.2):
    # creates file paths and loads the data
    path_x = data_path+"_val.csv"
    path_y = data_path+".csv"
    data_x = np.genfromtxt(path_x, delimiter=',', dtype='float32')
    data_y = np.transpose(np.genfromtxt(path_y, delimiter=',', dtype='float32'))
    # normalizes the data (use these values to normalize any future data)
    mean, std = data_x.mean(), data_x.std()
    stats = {'mean':float(mean), 'std':float(std)}
    data_x = (data_x-mean)/std
    # splits the data into train/test groups, converts to tensors
    train_x, test_x, train_y, test_y = train_test_split(data_x, data_y,test_size=float(percent_test),random_state=RANDOM_SEED)
    val_x, test_x, val_y, test_y = train_test_split(test_x, test_y,test_size=0.5,random_state=RANDOM_SEED)
    train_x = tf.convert_to_tensor(train_x, dtype=tf.float32)
    val_x = tf.convert_to_tensor(val_x, dtype=tf.float32)
    test_x = tf.convert_to_tensor(test_x, dtype=tf.float32)
    train_y = tf.convert_to_tensor(train_y, dtype=tf.float32)
    val_y = tf.convert_to_tensor(val_y, dtype=tf.float32)
    test_y = tf.convert_to_tensor(test_y, dtype=tf.float32)
    return train_x, val_x, test_x, train_y, val_y, test_y, stats

def plot_vs_epoch(name, output_folder, values1, values2=None, compare=False, display=False):
    path = str(output_folder + '/figures')
    name =  name.capitalize()
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
    return

def save_json(data, name, output_folder):
    file_name = str(output_folder + '/' + name + '.json')
    with open(file_name, 'w') as f:
        json.dump(data, f, indent=4)
    print("saved " + str(name) + '.json at ' + str(output_folder))
    return

def train_model(train_x, val_x, test_x, train_y, val_y, test_y, config):
    # Setting variables/initialization
    input_shape, output_shape = train_x.shape, train_y.shape
    initializer = glorot_normal(RANDOM_SEED)

    # Model definition
    model = Sequential()
    # First layer
    model.add(Input(shape=(input_shape[1])))
    # Hidden layers
    for i in range(config['num_hidden']):
        nodes_call = 'num_nodes' + str(i)
        dropout_call = 'dropout_rate' + str(i)
        model.add(Dense(
            config[nodes_call], 
            activation='relu', 
            kernel_initializer=initializer,
        ))
        model.add(Dropout(config[dropout_call]))
    # Output layer
    model.add(Dense(output_shape[1]))
    model.compile(
        optimizer=Adam(learning_rate=config['lr']), 
        loss=MeanSquaredError(), 
        metrics=METRICS,
    )
    print(model.summary())
    print("Starting training")
    # Train model
    start_time = time.time()
    history = model.fit(
        train_x, 
        train_y, 
        epochs=config['num_epochs'],
        batch_size=config['batch_size'], 
        validation_data=(val_x, val_y), 
        verbose=2,
    )
    scores = model.evaluate(test_x, test_y, return_dict=True)
    print("Finished training")
    print("Training time: " + str(time.time()-start_time))

    return model, history, scores

def main(config, data_path, output_folder):
    # Clear existing output
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    
    # Load data
    train_x, val_x, test_x, train_y, val_y, test_y, stats = load_data(data_path)

    # Train model
    model, history, scores = train_model(train_x, val_x, test_x, train_y, val_y, test_y, config)

    # Save training info 
    model.save(os.path.join(output_folder, 'model'))
    save_json(history.history, 'histories', output_folder)
    save_json(stats, 'stats', output_folder)
    save_json(scores, 'scores', output_folder)
    for key in history.history:
        plot_vs_epoch(key, output_folder, history.history[key])
        
    # Plot example prediction
    input_val = tf.reshape(test_x[0], shape=[1, test_x.shape[1]])
    predicted = model.predict(input_val).flatten()
    actual = test_y[0].numpy().flatten()
    plot_vs_epoch('Cross sectional scattering', output_folder, predicted, actual, compare=True, display=True)
    return model

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Physics Net Training")
    parser.add_argument("--data_path",type=str,default='data/8_layer_tio2') # Where the data file is. Note: This assumes a file of _val.csv and .csv 
    parser.add_argument("--output_folder",type=str,default='results/test1') #Where to output the results to. Note: No / at the end. 

    args = parser.parse_args()
    dict = vars(args)

    # Hyperparameters (used for tuning)
    # config = {
    #     'lr': tune.uniform(0.0001,0.1),
    #     'batch_size': tune.qrandint(16, 128, 16),
    #     'num_epochs': 10,
    #     'num_hidden': tune.randint(1,5), # the upper limit is exclusive, so options are (1,2,3,4)
    #     'num_nodes0': tune.qrandint(64, 320, 64),
    #     'num_nodes1': tune.qrandint(64, 320, 64),
    #     'num_nodes2': tune.qrandint(64, 320, 64),
    #     'num_nodes3': tune.qrandint(64, 320, 64),
    #     'dropout_rate0': tune.quniform(0.3, 0.8, 0.1),
    #     'dropout_rate1': tune.quniform(0.3, 0.8, 0.1),
    #     'dropout_rate2': tune.quniform(0.3, 0.8, 0.1),
    #     'dropout_rate3': tune.quniform(0.3, 0.8, 0.1),
    # }

    config = {
        'lr': 0.001,
        'batch_size': 64,
        'num_epochs': 100,
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
        
    kwargs = {  
        'data_path':dict['data_path'],
        'output_folder':dict['output_folder'],
    }

    main(config, **kwargs)