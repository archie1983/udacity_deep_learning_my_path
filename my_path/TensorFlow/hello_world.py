import tensorflow as tf

# Create TensorFlow object called hello_constant
hello_constant = tf.constant('Hello World!')

with tf.Session() as sess:
    # Run the tf.constant operation in the session
    output = sess.run(hello_constant)
    print(output)

# A is a 0-dimensional int32 tensor
A = tf.constant(1234) 
# B is a 1-dimensional int32 tensor
B = tf.constant([123,456,789]) 
# C is a 2-dimensional int32 tensor
C = tf.constant([ [123,456,789], [222,333,444] ])

### A "TensorFlow Session", as shown above, is an environment for running a graph. 
### The session is in charge of allocating the operations to GPU(s) and/or CPU(s), 
### including remote machines. Let’s see how you use it.

with tf.Session() as sess:
    output = sess.run(hello_constant)
    print(output)

### The code has already created the tensor, hello_constant, from the previous lines. 
### The next step is to evaluate the tensor in a session.

### The code creates a session instance, sess, using tf.Session. The sess.run() function then 
### evaluates the tensor and returns the results.


### tf.placeholder()

### Sadly you can’t just set x to your dataset and put it in TensorFlow, because over time you'll 
### want your TensorFlow model to take in different datasets with different parameters. 
### You need tf.placeholder()!

### tf.placeholder() returns a tensor that gets its value from data passed to the tf.session.run() 
### function, allowing you to set the input right before the session runs.
### Session’s feed_dict

x = tf.placeholder(tf.string)

with tf.Session() as sess:
    output = sess.run(x, feed_dict={x: 'Hello World'})

### Use the feed_dict parameter in tf.session.run() to set the placeholder tensor. 
### The above example shows the tensor x being set to the string "Hello, world". 
### It's also possible to set more than one tensor using feed_dict as shown below.

x = tf.placeholder(tf.string)
y = tf.placeholder(tf.int32)
z = tf.placeholder(tf.float32)

with tf.Session() as sess:
    output = sess.run(x, feed_dict={x: 'Test String', y: 123, z: 45.67})

print(output)

tf.subtract(tf.cast(tf.constant(2.0), tf.int32), tf.constant(1))   # 1

