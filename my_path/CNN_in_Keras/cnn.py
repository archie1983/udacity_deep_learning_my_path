from keras.layers import Conv2D

filters = 4 # number of filters
kernel_size = (3, 4) # the size of the convolution window. h = 3, w = 4
strides = 3 # stride of the convolution (default = 1)
padding = 'same' # we'll be padding the edges so that we don't ignore 
                 # them if kernel size goes over the edge (default = 'valid')
activation = 'relu' # Activation function (no activation applied by default)
input_shape = (45, 90, 1) # for the very first layer after the inputs, we need
                       # to specify input_shape. Here h = 45 and w = 90, so a
                       # picture of 45px * 90px. Third parameter in the tuple
                       # is the depth for 3D tensor inputs (e.g. RGB pictures)
                       # DO NOT INCLUDE input_shape if this is NOT the FIRST 
                       # HIDDEN LAYER.

model = keras.Sequential()

model.add(Conv2D(filters, kernel_size, strides, padding, activation='relu', input_shape))


