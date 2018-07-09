#The image below is an example of a convolution with a 3x3 filter and a stride of 1.

#The convolution for each 3x3 section is calculated against the weight, [[1, 0, 1], [0, 1, 0], [1, 0, 1]], and then a bias is added to create the convolved feature on the right. In this case, the bias is zero.

#Convolutional Layers in TensorFlow
#Let's examine how to implement a convolutional layer in TensorFlow.

#TensorFlow provides the tf.nn.conv2d(), tf.nn.bias_add(), and tf.nn.relu() functions to create your own convolutional layers.

# output depth
k_output = 64

# image dimensions
image_width = 10
image_height = 10
color_channels = 3

# convolution filter dimensions
filter_size_width = 5
filter_size_height = 5

# input/image
input = tf.placeholder(
    tf.float32,
    shape=[None, image_height, image_width, color_channels])

# weight and bias
weight = tf.Variable(tf.truncated_normal(
    [filter_size_height, filter_size_width, color_channels, k_output]))
bias = tf.Variable(tf.zeros(k_output))

# apply convolution
conv_layer = tf.nn.conv2d(input, weight, strides=[1, 2, 2, 1], padding='SAME')
# add bias
conv_layer = tf.nn.bias_add(conv_layer, bias)
# apply activation function
conv_layer = tf.nn.relu(conv_layer)
#The code above uses the tf.nn.conv2d() function to compute the convolution with weight as the filter and [1, 2, 2, 1] for the strides.

#TensorFlow uses a stride for each input dimension, [batch, input_height, input_width, input_channels].
#We generally always set the stride for batch and input_channels (i.e. the first and fourth element in the strides array) to be 1. This ensures that the model uses all batches and input channels. (It's good practice to remove the batches or channels you want to skip from the data set rather than use a stride to skip them.)
#You'll focus on changing input_height and input_width (while setting batch and input_channels to 1). The input_height and input_width strides are for striding the filter over input. This example code uses a stride of 2 with 5x5 filter over input. I've mentioned stride as one number because you usually have a square stride where height = width. When someone says they are using a stride of 2, they usually mean tf.nn.conv2d(x, W, strides=[1, 2, 2, 1]).
#The tf.nn.bias_add() function adds a 1-d bias to the last dimension in a matrix. (Note: using tf.add() doesn't work when the tensors aren't the same shape.)

#The tf.nn.relu() function applies a ReLU activation function to the layer.
