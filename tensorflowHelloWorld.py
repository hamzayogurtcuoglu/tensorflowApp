import tensorflow as tf

a = tf.constant([2])
b = tf.constant([3])
c = tf.add(a,b)
#session = tf.Session()
with tf.Session() as session:
    result = session.run(c)
    print(result)
#session.close()
#bole her defasinda kapatmak yerine with block yap
