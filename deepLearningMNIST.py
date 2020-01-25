import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

'''
0-Input MNIST dataset
1-Convolutional and Max-Pooling
2-Convolutional and Max-Pooling
3-Fully Connected Layer
4-Processing - Dropout
5-Readout Layer - Fully Connected
6-Output - Classified Digits
'''

sess = tf.InteractiveSession()

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

width = 28
height = 28
flat = width * height
class_output = 10

x = tf.placeholder(tf.float32,shape=[None,flat])
y_ =tf.placeholder(tf.float32,shape=[None,class_output])

def weight_variable(shape):
	initial = tf.truncated_normal(shape,stddev=0.1)
	return tf.Variable(initial)


def bias_variable(shape):
	initial = tf.constant(0.1,shape=shape)
	return tf.Variable(initial)

def conv2d(x,W):
	return tf.nn.conv2d(x,W,strides = [1,1,1,1],padding = 'SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')	

W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32]) #need 32 biases for 32 outputs
x_image =tf.reshape(x,[-1,28,28,1])
convolve1 = conv2d(x_image,W_conv1) + b_conv1
h_conv1 = tf.nn.relu(convolve1) #activation function
h_pool1 = max_pool_2x2(h_conv1)
layer1 = h_pool1

#filter kernel 5x5 25pixel ,input channels 32 (we had 32 feature maps),64 output feature maps 
W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64]) #need 64 biases for 64 outputs
convolve2 = conv2d(layer1,W_conv2) + b_conv2
h_conv2 = tf.nn.relu(convolve2) #activation function
h_pool2 = max_pool_2x2(h_conv2)
layer2 = h_pool2


#Fully Connected Layer (Softmax)
W_fc1 = weight_variable([7*7*64,1024])
b_fc1 = bias_variable([1024])
layer2_matrix = tf.reshape(layer2,[-1,7*7*64])
matmul_fc1 = tf.matmul(layer2_matrix,W_fc1) + b_fc1
h_fc1 = tf.nn.relu(matmul_fc1)
layer3 = h_fc1


keep_prob =	tf.placeholder(tf.float32)
layer3_drop = tf.nn.dropout(layer3,keep_prob)


W_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])
matmul_fc2 = tf.matmul(layer3_drop,W_fc2) + b_fc2
y_conv = tf.nn.softmax(matmul_fc2)
layer4 = y_conv


#Define Loss Function 
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(layer4),reduction_indices=[1]))
#Define Optimizer
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
#Define Accuracy
correct_prediction = tf.equal(tf.argmax(layer4,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

sess.run(tf.initialize_all_variables())


for i in range(1100):
	batch = mnist.train.next_batch(50)
	if i%100 == 0:
		train_accuracy = accuracy.eval(feed_dict={x:batch[0],y_:batch[1],keep_prob:1.0})			
	train_step.run(feed_dict={x:batch[0],y_:batch[1],keep_prob:0.5})


for i in range(20000):
	batch = mnist.train.next_batch(50)
	if i%100 == 0:
		train_accuracy = accuracy.eval(feed_dict={x:batch[0],y_:batch[1],keep_prob:1.0})			
		print("step %d ,training accuracy %g"%(i,train_accuracy))
	train_step.run(feed_dict={x:batch[0],y_:batch[1],keep_prob:0.5})


print("test accuracy %g"%accuracy.eval(feed_dict={x:mnist.test.images,y_:mnist.test.labels})*100)
sess.close()
