import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorlayer as tl
from keras.preprocessing.image import array_to_img

#
batch_size = 128
noise_size = 100
img_height, img_width = 32, 32
channels = 1
n_epochs = 20000
k1_steps = 1
k2_steps = 2

data = pd.read_csv('train.csv')
data_y = data['label'].values.astype(np.int64)
data_X = data.drop('label', axis=1).values
data_X = data_X.reshape(-1, 28, 28, 1)
data_X = np.pad(data_X, [[0, 0], [2, 2], [2, 2], [0, 0]], 'constant')
# data_X = data_X / 255.
data_X = -1.0 + (data_X - 0.0) / 255.0 * 2.0


class DCGAN():

    def __init__(self, sess, img_height, img_width, channels, noise_size, batch_size):
        self._img_height = img_height
        self._img_width = img_width
        self._channels = channels
        self._noise_size = noise_size
        self._batch_size = batch_size
        self._model_file = 'model.npz'
        self._sess = sess
        self._graph = self._sess.graph
        self._trained = False
        with self._graph.as_default():
            self._g_inputs = tf.placeholder(tf.float32, [self._batch_size, self._noise_size], name='z_noise')
            self._d_inputs = tf.placeholder(tf.float32,
                                            [self._batch_size, self._img_height, self._img_width, self._channels],
                                            name='img_input')
            # y_ = tf.placeholder(tf.int64, [None], name='y_')
            # definle network
            self._gen_network = self._img_generator()
            self._dis_network = self._img_discriminator()
            self._dis_gen_network = self._img_discriminator(is_train=True, reuse=True, input_network=self._gen_network)
            self._d_params = self._dis_network.all_params
            self._g_params = self._gen_network.all_params
            self._all_params = self._dis_gen_network.all_params
            with tf.device('/gpu:0'):
                # define loss
                self._d_loss = self._loss(self._dis_network) + self._loss(self._dis_gen_network, False)
                self._g_loss = self._loss(self._dis_gen_network, False)

                # define training ops
                self._d_train = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5)\
                                  .minimize(-self._d_loss, var_list=self._d_params)
                self._g_train = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5)\
                                  .minimize(self._g_loss, var_list=self._g_params)

    def _img_generator(self, is_train=True, reuse=False):
        with tf.variable_scope('generator', reuse=reuse):
            tl.layers.set_name_reuse(reuse)
            net_in = tl.layers.InputLayer(self._g_inputs, name='in')
            # [-1. 100] -> [-1, 8192]
            net_h0 = tl.layers.DenseLayer(net_in, n_units=4*4*512,
                                          W_init=tf.random_normal_initializer(stddev=0.02),
                                          b_init=tf.constant_initializer(0.0),
                                          act=tf.identity, name='h0/lin')
            # [-1, 8192] -> [-1, 4, 4, 512]
            net_h0 = tl.layers.ReshapeLayer(net_h0, shape=[-1, 4, 4, 512],
                                            name='h0/reshape')
            net_h0 = tl.layers.BatchNormLayer(net_h0, is_train=is_train,
                                              act=tf.nn.relu,
                                              name='h0/batch_norm')

            # [-1, 4, 4, 512] -> [-1, 8, 8, 256]
            net_h1 = tl.layers.DeConv2dLayer(net_h0,
                                             shape=[5, 5, 256, 512],
                                             output_shape=[batch_size, 8, 8, 256],
                                             strides=[1, 2, 2, 1],
                                             W_init=tf.random_normal_initializer(stddev=0.02),
                                             b_init=tf.constant_initializer(0.0),
                                             act=tf.identity, name='h1/decon2d')
            net_h1 = tl.layers.BatchNormLayer(net_h1, is_train=is_train,
                                              act=tf.nn.relu,
                                              name='h1/batch_norm')

            # [-1, 8, 8, 256] -> [-1, 16, 16, 128]
            net_h2 = tl.layers.DeConv2dLayer(net_h1,
                                             shape=[5, 5, 128, 256],
                                             output_shape=[batch_size, 16, 16, 128],
                                             strides=[1, 2, 2, 1],
                                             W_init=tf.random_normal_initializer(stddev=0.02),
                                             b_init=tf.constant_initializer(0.0),
                                             act=tf.identity, name='h2/decon2d')
            net_h2 = tl.layers.BatchNormLayer(net_h2, is_train=is_train,
                                              act=tf.nn.relu,
                                              name='h2/batch_norm')

            # [-1, 16, 16, 128] -> [-1, 32, 32, channels]
            net_h3 = tl.layers.DeConv2dLayer(net_h2,
                                             shape=[5, 5, channels, 128],
                                             output_shape=[batch_size, img_height, img_width, channels],
                                             strides=[1, 2, 2, 1],
                                             W_init=tf.random_normal_initializer(stddev=0.02),
                                             b_init=tf.constant_initializer(0.0),
                                             act=tf.nn.tanh,
                                             name='h3/decon2d')
        return net_h3

    def _img_discriminator(self, is_train=True, reuse=False, input_network=None):
        with tf.variable_scope('discriminator', reuse=reuse):
            tl.layers.set_name_reuse(reuse)
            if input_network is None:
                network = tl.layers.InputLayer(self._d_inputs, name='d_input_layer')
            else:
                network = input_network
            # [-1, 32, 32, 1] -> [-1, 16, 16, 32]
            network = tl.layers.Conv2dLayer(layer=network,
                                            act=tf.identity,
                                            shape=[3, 3, 1, 32],
                                            strides=[1, 2, 2, 1],
                                            padding='SAME',
                                            name='conv_1')
            network = tl.layers.BatchNormLayer(layer=network,
                                               act=lambda x: tl.act.lrelu(x, 0.2),
                                               is_train=is_train,
                                               name='batch_norm_1')
            # [-1, 16, 16, 32] -> [-1, 8, 8, 64]
            network = tl.layers.Conv2dLayer(layer=network,
                                            act=tf.identity,
                                            shape=[3, 3, 32, 64],
                                            strides=[1, 2, 2, 1],
                                            padding='SAME',
                                            name='conv_2')
            network = tl.layers.BatchNormLayer(layer=network,
                                               act=lambda x: tl.act.lrelu(x, 0.2),
                                               is_train=is_train,
                                               name='batch_norm_2')
            # flatten [-1, 8, 8, 64] -> [-1, 4096]
            network = tl.layers.FlattenLayer(layer=network, name='flatten_layer')
            network = tl.layers.DropoutLayer(layer=network,
                                             keep=0.5, name='dropout_1')
            network = tl.layers.DenseLayer(layer=network,
                                           n_units=128,
                                           act=tf.identity,
                                           name='dense_1')
            network = tl.layers.BatchNormLayer(layer=network,
                                               act=lambda x: tl.act.lrelu(x, 0.2),
                                               is_train=is_train,
                                               name='batch_norm_3')
            network = tl.layers.DropoutLayer(layer=network, keep=0.8, name='dropout_2')
            network = tl.layers.DenseLayer(layer=network, n_units=1,
                                           act=tf.identity, name='output_layer')
            return network

    def _loss(self, network=None, is_dis=True):
        outputs = network.outputs
        if is_dis:
            prob = tf.nn.sigmoid(outputs)
        else:
            prob = 1 - tf.nn.sigmoid(outputs)
        return tf.reduce_mean(tf.log(prob))

    def train(self, data, n_epochs, k_steps=2, printable=True):
        self._sess.run(tf.initialize_all_variables())
        if os.path.exists(self._model_file):
            load_params = tl.files.load_npz(path='', name=self._model_file)
            tl.files.assign_params(self._sess, load_params, self._dis_gen_network)
        else:
            for n_epoch in range(n_epochs):
                # update discriminator
                x = data[np.random.choice(data.shape[0], size=self._batch_size)]
                z = np.random.uniform(size=(self._batch_size, self._noise_size))
                feed_dict = {self._d_inputs: x, self._g_inputs: z}
                feed_dict.update(self._dis_gen_network.all_drop)
                feed_dict.update(self._dis_network.all_drop)
                _, d_loss = self._sess.run([self._d_train, self._d_loss], feed_dict=feed_dict)
                # update generator
                for _ in range(k_steps):
                    z = np.random.uniform(size=(self._batch_size, self._noise_size))
                    feed_dict = {self._g_inputs: z}
                    feed_dict.update(self._dis_gen_network.all_drop)
                    _, g_loss = self._sess.run([self._g_train, self._g_loss], feed_dict=feed_dict)
                if n_epoch % 100 == 0:
                    print('n_epoch: {}, d_loss: {}, g_loss: {}'.format(n_epoch, d_loss, g_loss))
                if n_epoch % 1000 == 0:
                    tl.files.save_npz(self._all_params, name='model.npz', sess=self._sess)
        self._trained = True

    def generate_img(self, n_nums=20, smooth=False):
        # generate images.
        if not self._trained:
            raise Exception('model is not trainded!')
        z = np.random.uniform(size=(self._batch_size, self._noise_size))
        feed_dict = {self._g_inputs: z}
        feed_dict.update(tl.utils.dict_to_one(self._gen_network.all_drop))
        outputs = self._sess.run(self._gen_network.outputs, feed_dict=feed_dict)
        for i in range(n_nums):
            img = array_to_img(outputs[i])
            img.save('gen_img/gen{}.jpg'.format(i))


if __name__ == '__main__':
    sess = tf.Session()
    dcgan = DCGAN(sess, img_height, img_width, channels, noise_size, batch_size)
    dcgan.train(data_X, 20000)
    dcgan.generate_img()
