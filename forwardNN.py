# Written and tested in Python 3.10.4

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.initializers import glorot_normal
import keras_tuner as kt
from tensorboard.plugins.hparams import api as hp
import os
import time
import json
import argparse
import shutil

# Initialize Constants
RANDOM_SEED = 42
METRICS = ['accuracy', 'MeanAbsolutePercentageError']

# Hyperparameters


# HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([16, 32]))
# HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.1, 0.2))
# HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd']))


# Helper Functions
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

def plot_vs_epoch(name, output_folder, values1, values2=None, compare=False):
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
    plt.show()
    return

def save_json(data, name, output_folder):
    file_name = str(output_folder + '/' + name + '.json')
    with open(file_name, 'w') as f:
        json.dump(data, f, indent=4)
    print("saved " + str(name) + '.json at ' + str(output_folder))
    return

def set_folder(output_folder):
    print("You can either delete the current results at " + str(output_folder) + ' or you can create new folder to save the data')
    choice = input("Type \'delete\' to delete or \'new\' to create a new folder: ")
    while True:
        if choice == 'delete':
            shutil.rmtree(output_folder)
            break
        elif choice == 'new':
            new_path = input("Please type in the new folder name: ")
            output_folder = new_path
            if os.path.exists(output_folder):
                print("path already exists, pick a new one")
                break
            else:
                break
    return output_folder

# Forward Training
# rewrite with hyperparameters 



# def build_forward(hp):
#     model = Sequential()
#     num_layers = hp,Int('num_layers', min_value = 2, max_value = 5)
#     hp_units1 = hp.Int('units1', min_value=32, max_value=512, step=32)


#     return model




def define_forward_model(input_shape, output_shape, lr_rate, lr_decay):
    # metric_list provides the metrics tracked and saved
    model = Sequential()
    initializer=glorot_normal(RANDOM_SEED)
    # add model layers
    # input layer
    model.add(Input(shape=(input_shape[1])))
    # hidden layers
    model.add(Dense(256, activation='relu', kernel_initializer=initializer))
    model.add(Dense(256, activation='relu', kernel_initializer=initializer))
    model.add(Dense(256, activation='relu', kernel_initializer=initializer))
    # output layer
    model.add(Dense(output_shape[1]))
    model.compile(optimizer=Adam(learning_rate=lr_rate), loss=MeanAbsoluteError(), metrics=METRICS)
    print(model.summary())
    return model

def evaluate_forward(train_x, val_x, test_x, train_y, val_y, test_y, batch_size, num_epochs, lr_rate, lr_decay, percent_val):
    # Assume that input and output are 1D arrays, if not, adjust as needed
    input_shape, output_shape = train_x.shape, train_y.shape
    end_step = np.ceil(input_shape[0]*(1-percent_val))*num_epochs
    model = define_forward_model(input_shape, output_shape, lr_rate, lr_decay)
    history = model.fit(train_x, train_y, epochs=num_epochs, batch_size=batch_size, validation_data=(val_x, val_y), verbose=1)
    scores = model.evaluate(test_x, test_y, return_dict=True)
    return model, history, scores

def train_forward(data_path, output_folder, batch_size, num_epochs, lr_rate, lr_decay, num_layers, n_hidden, percent_val, patienceLimit):
    # Set output folder
    # shutil.rmtree(output_folder) 
    if os.path.exists(output_folder):
        output_folder = set_folder(output_folder)
    train_x, val_x, test_x, train_y, val_y, test_y, stats = load_data(data_path, percent_val)
    print("Beginning training")
    model, history, scores = evaluate_forward(train_x, val_x, test_x, train_y, val_y, test_y, batch_size, num_epochs, lr_rate, lr_decay, percent_val)
    print("Completed training")
    print(model.summary())
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
    plot_vs_epoch('Cross sectional scattering', output_folder, predicted, actual, compare=True)
    return model






# Main
def main(data_path, output_folder, batch_size, num_epochs, lr_rate, lr_decay, num_layers, n_hidden, percent_val, patienceLimit):
    model = train_forward(data_path, output_folder, batch_size, num_epochs, lr_rate, lr_decay, num_layers, n_hidden, percent_val, patienceLimit)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Physics Net Training")
    parser.add_argument("--data_path",type=str,default='data/8_layer_tio2') # Where the data file is. Note: This assumes a file of _val.csv and .csv 
    parser.add_argument("--output_folder",type=str,default='test_results1') #Where to output the results to. Note: No / at the end. 
    parser.add_argument("--batch_size",type=int,default=64) # Batch Size
    parser.add_argument("--num_epochs",type=int,default=1000) #Max number of epochs to consider at maximum, if patience condition is not met. 
    parser.add_argument("--lr_rate",type=float,default=.001) # Learning Rate. 
    parser.add_argument("--lr_decay",type=float,default=.999) # Learning rate decay. It decays by this factor every epoch.
    parser.add_argument("--num_layers",type=int,default=4) # Number of layers in the network. 
    parser.add_argument("--n_hidden",type=int,default=225) # Number of neurons per layer. Fully connected layers. 
    parser.add_argument("--percent_val",type=float,default=.2) # Amount of the data to split for validation/test. The validation/test are both split equally. 
    parser.add_argument("--patience",type=int,default=10) # Patience for stopping. If validation loss has not decreased in this many steps, it will stop the training. 

    args = parser.parse_args()
    dict = vars(args)
    print(dict)
        
    kwargs = {  
            'data_path':dict['data_path'],
            'output_folder':dict['output_folder'],
            'batch_size':dict['batch_size'],
            'num_epochs':dict['num_epochs'],
            'lr_rate':dict['lr_rate'],
            'lr_decay':dict['lr_decay'],
            'num_layers':dict['num_layers'],
            'n_hidden':dict['n_hidden'],
            'percent_val':dict['percent_val'],
            'patienceLimit':dict['patience'],
            }
    main(**kwargs)