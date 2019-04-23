import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
#from matplotlib import pyplot as plt

def soft_mnist_model(path):
    mnist = input_data.read_data_sets(path,one_hot=True)
    
    '''
    for i in range(10):
        image = mnist.train.images[i].reshape(28,28)
        plt.imshow(image)
        plt.show()
        print("类别为：{}".format(mnist.train.labels[i]))
    '''
        
    x = tf.placeholder(tf.float32,[None,784])
    w = tf.Variable(tf.zeros([784,10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.nn.softmax(tf.matmul(x,w)+b)
    y_ = tf.placeholder(tf.float32,[None,10])
    
    # 交叉熵损失
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y)))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    for i in range(1000):
        batch_xs,batch_ys = mnist.train.next_batch(100)
        sess.run(train_step,feed_dict={x:batch_xs,y_:batch_ys})
        
    correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
    accuracy= tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    print("在测试集上的准确率：{}".format(sess.run(accuracy,feed_dict={x:mnist.test.images,y_:mnist.test.labels})))
    
if __name__ == "__main__":
    path = "D:/Program Files/PythonProject/tensorflow/mnist_data/"
    soft_mnist_model(path)