# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# Create your first MLP in Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, load_model
import numpy
#from google.colab import files
#uploaded = files.upload();

import pandas as pd
# fix random seed for reproducibility
seed = 7;
numpy.random.seed(seed);

dataset = pd.read_csv("D:/ExcelR/50_Startups/forestfires.csv");
dataset;
from sklearn import preprocessing;
label_encoder = preprocessing.LabelEncoder();
dataset["month"] = label_encoder.fit_transform(dataset["month"]);
dataset["day"] = label_encoder.fit_transform(dataset["day"]);
dataset["size_category"] = label_encoder.fit_transform(dataset["size_category"]);

dataset;

#dataset = numpy.loadtxt("C:\\Users\\UNME\\Downloads\\DS Course\\Neural Networks\\pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset.iloc[:,:11];
Y = dataset.iloc[:,-1];

X;
Y;
# create model
model = Sequential();
model.add(layers.Dense(50, input_dim=11,  activation='relu'));
model.add(layers.Dense(11,  activation='relu'));
model.add(layers.Dense(1, activation='sigmoid'));

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc']);

# Fit the model
history = model.fit(X, Y, validation_split=0.33, epochs=100, batch_size=10);

# evaluate the model
scores = model.evaluate(X, Y);
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100));

# Visualize training history

# list all data in history
history.history.keys();

# Visualize training history

# list all data in history
history.history.keys();

# summarize history for accuracy
import matplotlib.pyplot as plt
#%matplotlib inline;
plt.plot(history.history['acc']);
plt.plot(history.history['val_acc']);
plt.title('model accuracy');
plt.ylabel('accuracy');
plt.xlabel('epoch');
plt.legend(['train', 'test'], loc='upper left');
plt.show();
# summarize history for loss
plt.plot(history.history['loss']);
plt.plot(history.history['val_loss']);
plt.title('model loss');
plt.ylabel('loss');
plt.xlabel('epoch');
plt.legend(['train', 'test'], loc='upper left');
plt.show();