from keras.models import Sequential
from keras.layers import Conv2D

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=3, strides=2, padding='same', 
    activation='relu', input_shape=(128, 128, 3)))
model.summary()

### Number of parameters is: kernel_size_h * kernel_size_w * filters * depth + filters
### depth here is 1. Filters are added in the end, because they represent how many biases
### we will have (1 per filter)
