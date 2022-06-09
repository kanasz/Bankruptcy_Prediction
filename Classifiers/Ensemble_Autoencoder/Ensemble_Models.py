from tensorflow.keras.layers import Input,Dense,Dropout
from tensorflow.keras import regularizers
import tensorflow as tf

def create_ensemble_autoencoder_model_1_dynamic(input_dim=60,learning_rate=1e-3):
    encoding_dim = 0.5 * input_dim
    autoencoder = tf.keras.Sequential()
    autoencoder.add(Input(shape=(input_dim,)))
    autoencoder.add(Dense(encoding_dim, activation="relu", activity_regularizer=regularizers.l1(learning_rate)))
    autoencoder.add(Dense(input_dim, activation="linear"))
    return  autoencoder

def create_ensemble_autoencoder_model_2_dynamic(input_dim=60,learning_rate=1e-3):
    encoding_dim = 0.5 * input_dim
    autoencoder = tf.keras.Sequential()
    autoencoder.add(Input(shape=(input_dim,)))
    autoencoder.add(Dense(encoding_dim, activation="relu", activity_regularizer=regularizers.l1(learning_rate)))
    autoencoder.add(Dense(input_dim, activation="tanh"))
    return  autoencoder


def create_ensemble_autoencoder_model_16_dynamic(input_dim=60,learning_rate=1e-3):
    encoding_dim = 0.5 * input_dim
    hidden_dim = encoding_dim * 0.5
    if hidden_dim < 3:
        hidden_dim = 3
    autoencoder = tf.keras.Sequential()
    autoencoder.add(Input(shape=(input_dim,)))
    autoencoder.add(Dropout(0.25))
    autoencoder.add(Dense(encoding_dim, activation="relu", activity_regularizer=regularizers.l1(learning_rate)))
    autoencoder.add(Dropout(0.50))
    autoencoder.add(Dense(int(hidden_dim ), activation="sigmoid"))
    #autoencoder.add(Dense(int(hidden_dim), activation="sigmoid"))
    autoencoder.add(Dense(encoding_dim, activation="relu"))
    autoencoder.add(Dense(input_dim, activation="tanh"))
    return  autoencoder

def create_ensemble_autoencoder_model_16_dynamic_article(input_dim=60,learning_rate=1e-3):
    encoding_dim = 0.5 * input_dim
    hidden_dim = encoding_dim * 0.5
    if hidden_dim < 3:
        hidden_dim = 3
    autoencoder = tf.keras.Sequential()
    autoencoder.add(Input(shape=(input_dim,)))
    autoencoder.add(Dropout(0.25))
    autoencoder.add(Dense(encoding_dim, activation="relu", activity_regularizer=regularizers.l1(learning_rate)))
    autoencoder.add(Dropout(0.50))
    autoencoder.add(Dense(int(hidden_dim ), activation="tanh"))
    #autoencoder.add(Dense(int(hidden_dim), activation="sigmoid"))
    autoencoder.add(Dense(encoding_dim, activation="relu"))
    autoencoder.add(Dense(input_dim, activation="tanh"))
    return  autoencoder

def create_ensemble_autoencoder_model_17_dynamic(input_dim=60,learning_rate=1e-3):
    encoding_dim = 0.5 * input_dim
    hidden_dim = encoding_dim * 0.5
    if hidden_dim < 3:
        hidden_dim = 3
    autoencoder = tf.keras.Sequential()
    autoencoder.add(Input(shape=(input_dim,)))
    #autoencoder.add(Dropout(0.25))
    autoencoder.add(Dense(encoding_dim, activation="relu", activity_regularizer=regularizers.l1(learning_rate)))
    #autoencoder.add(Dropout(0.50))
    autoencoder.add(Dense(int(hidden_dim ), activation="tanh"))
    autoencoder.add(Dense(encoding_dim, activation="relu"))
    autoencoder.add(Dense(input_dim, activation="tanh"))
    return  autoencoder

def create_ensemble_autoencoder_model_18_dynamic(input_dim=60,learning_rate=1e-3):
    encoding_dim = 0.5 * input_dim
    hidden_dim = encoding_dim * 0.5
    if hidden_dim < 3:
        hidden_dim = 3
    autoencoder = tf.keras.Sequential()
    autoencoder.add(Input(shape=(input_dim,)))
    autoencoder.add(Dropout(0.25))
    autoencoder.add(Dense(encoding_dim, activation="relu", activity_regularizer=regularizers.l1(learning_rate)))
    autoencoder.add(Dropout(0.25))
    autoencoder.add(Dense(int(hidden_dim ), activation="relu"))
    autoencoder.add(Dropout(0.25))
    autoencoder.add(Dense(encoding_dim, activation="relu"))
    autoencoder.add(Dropout(0.25))
    autoencoder.add(Dense(input_dim, activation="tanh"))
    return  autoencoder

def create_ensemble_autoencoder_model_19_dynamic(input_dim=60,learning_rate=1e-3):
    encoding_dim = 0.7 * input_dim
    hidden_dim = encoding_dim * 0.7
    latent_dim = hidden_dim * 0.7
    if hidden_dim < 3:
        hidden_dim = 3
    if latent_dim < 3:
        hidden_dim = 3

    autoencoder = tf.keras.Sequential()
    autoencoder.add(Input(shape=(input_dim,)))
    autoencoder.add(Dropout(0.25))
    autoencoder.add(Dense(encoding_dim, activation="relu", activity_regularizer=regularizers.l1(learning_rate)))
    autoencoder.add(Dropout(0.50))
    autoencoder.add(Dense(int(hidden_dim ), activation="sigmoid"))
    autoencoder.add(Dense(int(latent_dim), activation="tanh"))
    autoencoder.add(Dense(int(hidden_dim), activation="sigmoid"))
    autoencoder.add(Dense(encoding_dim, activation="relu"))
    autoencoder.add(Dense(input_dim, activation="tanh"))
    return  autoencoder


def create_ensemble_autoencoder_model_24_dynamic(input_dim=60,learning_rate=1e-3):
    encoding_dim = 0.7 * input_dim
    autoencoder = tf.keras.Sequential()
    autoencoder.add(Input(shape=(input_dim,)))
    autoencoder.add(Dropout(0.25))
    autoencoder.add(Dense(int(encoding_dim ), activation="tanh"))
    autoencoder.add(Dense(int(encoding_dim), activation="tanh"))
    autoencoder.add(Dense(input_dim, activation="linear"))
    return  autoencoder

def create_ensemble_autoencoder_model_25_dynamic(input_dim=60,learning_rate=1e-3):
    encoding_dim = 0.7 * input_dim
    latent_dim = 0.7* encoding_dim
    if latent_dim < 3:
        latent_dim = 3
    autoencoder = tf.keras.Sequential()
    autoencoder.add(Input(shape=(input_dim,)))
    #autoencoder.add(Dropout(0.25))
    autoencoder.add(Dense(int(encoding_dim ), activation="relu"))
    autoencoder.add(Dense(int(latent_dim), activation="relu"))
    autoencoder.add(Dense(int(encoding_dim), activation="relu"))
    autoencoder.add(Dense(input_dim, activation="linear"))
    return  autoencoder

def create_ensemble_autoencoder_model_26_dynamic(input_dim=60,learning_rate=1e-3):
    encoding_dim = 0.5 * input_dim
    hidden_dim = encoding_dim * 0.5
    if hidden_dim < 3:
        hidden_dim = 3
    autoencoder = tf.keras.Sequential()
    autoencoder.add(Input(shape=(input_dim)))
    autoencoder.add(Dense(encoding_dim, activation="sigmoid", activity_regularizer=regularizers.l1(0.0001)))
    autoencoder.add(Dense(int(hidden_dim ), activation="sigmoid"))
    autoencoder.add(Dense(encoding_dim, activation="sigmoid"))
    autoencoder.add(Dense(input_dim, activation="tanh"))
    return  autoencoder

def create_ensemble_autoencoder_model_27_dynamic(input_dim=60,learning_rate=1e-3):
    encoding_dim = 0.5 * input_dim
    hidden_dim = encoding_dim * 0.5
    if hidden_dim < 3:
        hidden_dim = 3
    autoencoder = tf.keras.Sequential()
    autoencoder.add(Input(shape=(input_dim)))
    autoencoder.add(Dense(encoding_dim, activation="sigmoid", activity_regularizer=regularizers.l1(0.0001)))
    autoencoder.add(Dense(int(hidden_dim ), activation="sigmoid"))
    autoencoder.add(Dense(encoding_dim, activation="sigmoid"))
    autoencoder.add(Dense(input_dim, activation="linear"))
    return  autoencoder