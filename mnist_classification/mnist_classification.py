import tensorflow as tf
import os
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"


class MnistClassifier():
    def __init__(self, global_step):
        self.lr = tf.placeholder(dtype=tf.float32, shape=[], name='learning_rate')
        self.X = tf.placeholder(dtype=tf.float32, shape=[None, 784], name='image_X')
        self.Y = tf.placeholder(dtype=tf.float32, shape=[None, 10], name='label_Y')
        self.output, self.cost = self.build_model(self.X, self.Y)
        self.optim = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.train_op = self.optim.minimize(loss=self.cost)
        self.global_step = global_step
        self.prediction = tf.equal(tf.argmax(self.Y, 1), tf.argmax(self.output, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.prediction, tf.float32))

    def build_model(self, X, Y):
        net = tf.layers.dense(X, 100, activation='relu')
        net = tf.layers.dense(net, 100, activation='relu')
        net = tf.nn.dropout(net, keep_prob=0.8)
        net = tf.layers.dense(net, 50, activation='relu')
        net = tf.layers.dense(net, 50, activation='relu')
        net = tf.nn.dropout(net, keep_prob=0.8)
        net = tf.layers.dense(net, 25, activation='relu')
        net = tf.layers.dense(net, 25, activation='relu')
        net = tf.nn.dropout(net, keep_prob=0.8)
        net = tf.layers.dense(net, 10, activation='relu')

        cost = tf.losses.softmax_cross_entropy(onehot_labels=Y, logits=net)

        return net, cost

    def train(self, mnist):
        session_config = tf.ConfigProto(allow_soft_placement=True)
        session_config.gpu_options.allow_growth = True
        session = tf.Session(config=session_config)

        session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        avg_cost = 0
        train_acc = 0

        for i in range(self.global_step):
            batch_xs, batch_ys = mnist.train.next_batch(256)
            acc, cost, _ = session.run(fetches=[self.accuracy, self.cost, self.train_op],
                                  feed_dict={self.X: batch_xs, self.Y: batch_ys, self.lr: 0.001})
            avg_cost += cost
            train_acc += acc
            if (i+1) % 100 == 0:
                print('[Step %04d] avg loss: %.4f, train acc: %.4f' % (i+1, avg_cost/100, train_acc/100))
                avg_cost = 0
                train_acc = 0
            if (i+1) % 500 == 0:
                val_acc = session.run(self.accuracy,
                                      feed_dict={self.X: mnist.validation.images, self.Y: mnist.validation.labels})
                saver.save(session, '/disk3/hyeon/mnist_model/tr-%05d' % i)
                print('[Validation] Acc: %.4f' % val_acc)

        print('[Test] Acc: %.4f'
              % (session.run(self.accuracy, feed_dict={self.X: mnist.test.images, self.Y: mnist.test.labels})))

if __name__=='__main__':
    model = MnistClassifier(50000)
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    model.train(mnist)