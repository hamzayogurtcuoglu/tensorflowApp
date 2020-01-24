import numpy as np

x=[3,4,5]
h=[2,1,0]

y = np.convolve(x,h)
print(y)


x=[6,2]
h=[1,2,5,4]

y = np.convolve(x,h,"full")
print(y) 


x=[6,2]
h=[1,2,5,4]

y = np.convolve(x,h,"valid")
print(y) 


from scipy import signal as sg

I = [[255,7,3],
	[212,240,4],
	[218,216,230],]

g=[[-1,1]]

print("without zero padding")
print("{0} \n".format(sg.convolve(I,g,"valid")))

print("with zero padding")
print("{0} \n".format(sg.convolve(I,g)))



import tensorflow as tf

input  = tf.Variable(tf.random_normal([1,10,10,1]))
filter  = tf.Variable(tf.random_normal([3,3,1,1]))
op = tf.nn.conv2d(input,filter,strides=[1,1,1,1],padding = "VALID")
op2 = tf.nn.conv2d(input,filter,strides=[1,1,1,1],padding = "SAME")

init = tf.initialize_all_variables()

with tf.Session() as sess:
	
	sess.run(init)

	print("Input \n") 
	print("{0} \n".format(input.eval()))
	print("Filter/Kernal \n")
	print("{0} \n".format(filter.eval()))
	print("Result/Feature Map with valid positions \n")
	result = sess.run(op)
	print(result)
	print("\n")
	print("Result/Feature Map with padding \n")
	result2 = sess.run(op2)
	print(result2)

