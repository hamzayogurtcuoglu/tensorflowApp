import tensorflow as tf

###### Variable############
state = tf.Variable(0)
one = tf.constant(1)
new_value = tf.add(state,one)
update = tf.assign(state,new_value)
init_op = tf.initialize_all_variables()


with tf.Session() as session:
    session.run(init_op)
    print(session.run(state))
    for _ in range(3):
        session.run(update)
        print(session.run(state))


######## PlaceHolder ########

a = tf.placeholder(tf.float32)
b = a*2
with tf.Session() as sess:
    result = sess.run(b,feed_dict ={a:3.5})
    print(result)


dictionary = {a:[[1,2,3,4],[5,6,7,8]]}

with tf.Session() as sess:
    result = sess.run(b,feed_dict=dictionary)
    print(result)


