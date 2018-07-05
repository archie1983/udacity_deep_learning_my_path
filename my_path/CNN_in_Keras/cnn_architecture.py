## Just as with neural networks, we create a CNN in Keras by first creating a Sequential model.

from keras.models import Sequential
## We import several layers, including layers that are familiar from neural networks, and new layers that we learned about in this lesson.

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
## As with neural networks, we add layers to the network by using the .add() method:

model = Sequential()
model.add(Conv2D(filters=16, kernel_size=2, padding='same', activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=64, kernel_size=2, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Flatten())
model.add(Dense(500, activation='relu'))
model.add(Dense(10, activation='softmax'))
## The network begins with a sequence of three convolutional layers, followed by max pooling layers. These first six layers are designed to take the input array of image pixels and convert it to an array where all of the spatial information has been squeezed out, and only information encoding the content of the image remains. The array is then flattened to a vector in the seventh layer of the CNN. It is followed by two dense layers designed to further elucidate the content of the image. The final layer has one entry for each object class in the dataset, and has a softmax activation function, so that it returns probabilities.

## NOTE: In the video, you might notice that convolutional layers are specified with Convolution2D instead of Conv2D. Either is fine for Keras 2.0, but Conv2D is preferred.

## Things to Remember
## Always add a ReLU activation function to the Conv2D layers in your CNN. With the exception of the final layer in the network, Dense layers should also have a ReLU activation function.
## When constructing a network for classification, the final layer in the network should be a Dense layer with a softmax activation function. The number of nodes in the final layer should equal the total number of classes in the dataset.
## Have fun! If you start to feel discouraged, we recommend that you check out Andrej Karpathy's tumblr with user-submitted loss functions, corresponding to models that gave their owners some trouble. Recall that the loss is supposed to decrease during training. These plots show very different behavior :).
