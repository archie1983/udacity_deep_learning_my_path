import numpy as np
from keras.utils import np_utils
import tensorflow as tf

# Using TensorFlow 1.0.0; use tf.python_io in later versions
tf.python_io.control_flow_ops = tf

# Set random seed
np.random.seed(42)

# Our data
X = np.array([[0,0],[0,1],[1,0],[1,1]]).astype('float32')
y = np.array([[0],[1],[1],[0]]).astype('float32')

# Initial Setup for Keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation

# Building the model
xor = Sequential()

# Add required layers
# Set the first layer to a Dense() layer with an output width of 8 
# nodes and the input_dim set to the size of the training samples (in this case 2).
xor.add(Dense(8, input_dim=X.shape[1]))

# Add a tanh activation function.
xor.add(Activation('tanh'))

# Set the output layer width to 1, since the output has only two classes. 
# (We can use 0 for one class an 1 for the other)
xor.add(Dense(1))

# Use a sigmoid activation function after the output layer.
xor.add(Activation('sigmoid'))

# Specify loss as "binary_crossentropy", optimizer as "adam",
# and add the accuracy metric

### Once we have our model built, we need to compile it before it can be run. 
### Compiling the Keras model calls the backend (tensorflow, theano, etc.) and binds the optimizer, 
### loss function, and other parameters required before the model can be run on any input data. 
### We'll specify the loss function to be binary_crossentropy which can be used when there are only 
### two classes, and specify adam as the optimizer (which is a reasonable default when speed is a 
### priority). And finally, we can specify what metrics we want to evaluate the model with. 
### Here we'll use accuracy.
xor.compile(loss="binary_crossentropy", optimizer="adam", metrics = ["accuracy"])

# Uncomment this line to print the model architecture
xor.summary()

# Fitting the model
# Run the model for 50 epochs.
history = xor.fit(X, y, epochs=50, verbose=0)

# Scoring the model
score = xor.evaluate(X, y)
print("\nAccuracy: ", score[-1])

# Checking the predictions
print("\nPredictions:")
print(xor.predict_proba(X))
