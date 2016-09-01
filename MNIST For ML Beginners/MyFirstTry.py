'''
MNIST For ML Beginners

This tutorial is intended for readers who are new to both machine learning and 
TensorFlow.
https://www.tensorflow.org/versions/r0.10/tutorials/mnist/beginners/index.html

Created on Sep 1, 2016

@author: yxg383
'''

#Import data from The MNIST data
from tensorflow.examples.tutorials.mnist import input_data
minst = input_data.read_data_sets("MNIST_data/", one_hot=True)

'''
The MNIST data is split into three parts: 55,000 data points of training data 
(mnist.train), 10,000 points of test data (mnist.test), and 5,000 points of 
validation data (mnist.validation). This split is very important: it's essential 
in machine learning that we have separate data which we don't learn from so that 
we can make sure that what we've learned actually generalizes!
'''
#Implementing the Regression
import tensorflow as tf
x = tf.placeholder(tf.float32,[None, 784])
"""
x isn't a specific value. It's a placeholder, a value that we'll input when we 
ask TensorFlow to run a computation. We want to be able to input any number of 
MNIST images, each flattened into a 784-dimensional vector. We represent this 
as a 2-D tensor of floating-point numbers, with a shape [None, 784]. (Here None 
means that a dimension can be of any length.)
"""
#the weights and biases for our model
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
'''
Notice that W has a shape of [784, 10] because we want to multiply the 
784-dimensional image vectors by it to produce 10-dimensional vectors of 
evidence for the difference classes. b has a shape of [10] so we can add it to 
the output.
'''
#implement our model
y = tf.nn.softmax(tf.matmul(x, W) + b)

#Training
'''
To implement cross-entropy we need to first add a new placeholder to input the 
correct answers
'''
y_ = tf.placeholder(tf.float32, [None, 10])

#implement the cross-entropy function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
"""
First, tf.log computes the logarithm of each element of y. Next, we multiply 
each element of y_ with the corresponding element of tf.log(y). Then 
tf.reduce_sum adds the elements in the second dimension of y, due to the 
reduction_indices=[1] parameter. Finally, tf.reduce_mean computes the mean over 
all the examples in the batch.
"""
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
'''
 minimize cross_entropy using the gradient descent algorithm with a learning 
 rate of 0.5. 
'''
#set up model
init = tf.initialize_all_variables()
# launch the model in a Session
sess = tf.Session()
sess.run(init)
#run the training step 1000 times
for i in range(1000):
  batch_xs, batch_ys = minst.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

#Evaluating Our Model
'''
That gives us a list of booleans. To determine what fraction are correct, we 
cast to floating point numbers and then take the mean. For example, 
[True, False, True, True] would become [1,0,1,1] which would become 0.75.
'''
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('The Accuracy: ' + str(sess.run(accuracy, feed_dict={x: minst.test.images, y_: minst.test.labels})))
