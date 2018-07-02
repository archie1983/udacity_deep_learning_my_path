import nn_model as nnm
import tensorflow as tf

saver = tf.train.Saver()

# Launch the graph
with tf.Session() as sess:
    saver.restore(sess, nnm.save_file)

    test_accuracy = sess.run(
        nnm.accuracy,
        feed_dict={nnm.features: nnm.mnist.test.images, nnm.labels: nnm.mnist.test.labels})

print('Test Accuracy: {}'.format(test_accuracy))
