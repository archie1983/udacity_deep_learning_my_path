from keras.models import Sequential
from keras.layers import Conv2D

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=3, strides=2, padding='same', 
    activation='relu', input_shape=(128, 128, 3)))
model.summary()

### Number of parameters is: kernel_size_h * kernel_size_w * filters * depth + filters
### depth here is 1. Filters are added in the end, because they represent how many biases
### we will have (1 per filter)


### Formula: Shape of a Convolutional Layer
### The shape of a convolutional layer depends on the supplied values of kernel_size, input_shape, padding, and stride. Let's define a few variables:

### K - the number of filters in the convolutional layer
### F - the height and width of the convolutional filters
### S - the stride of the convolution
### H_in - the height of the previous layer
### W_in - the width of the previous layer
### Notice that K = filters, F = kernel_size, and S = stride. Likewise, H_in and W_in are the first and second value of the input_shape tuple, respectively.

### The depth of the convolutional layer will always equal the number of filters K.

### If padding = 'same', then the spatial dimensions of the convolutional layer are the following:

### height = ceil(float(H_in) / float(S))
### width = ceil(float(W_in) / float(S))
### If padding = 'valid', then the spatial dimensions of the convolutional layer are the following:

### height = ceil(float(H_in - F + 1) / float(S))
### width = ceil(float(W_in - F + 1) / float(S))
