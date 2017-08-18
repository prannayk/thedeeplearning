import tensorflow as tf
import numpy as np
import scipy.misc
import sys
import time
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

class EBGAN():

	def batch_normalize(self, X, eps=1e-6,flag=False):
		if flag : 
			if X.get_shape().ndims == 4:
				mean, vari = tf.nn.moments(X, [0,1,2], keep_dims=True)
				return tf.nn.batch_normalization(X,mean, vari, variance_epsilon=eps)
			elif X.get_shape().ndims == 2:
				mean, vari = tf.nn.moments(X, 0, keep_dims=True)
				return tf.nn.batch_normalization(X, mean, vari, variance_epsilon=eps)
		if X.get_shape().ndims == 4 :
			mean = tf.reduce_mean(X,[0,1,2])
			stddev = tf.reduce_mean(tf.square(X-mean),[0,1,2])
			X = (X - mean)/tf.sqrt(stddev + eps)
		elif X.get_shape().ndims == 2:
			mean = tf.reduce_mean(X,[0])
			stddev = tf.reduce_mean(tf.square(X-mean),[0])
			X = (X - mean)/tf.sqrt(stddev + eps)
		else:
			raise NoImplementationForSuchDimensions
		return X

	def lrelu(self, X):
		return LeakyRelu(X)

	def generate(self, embedding, classes, scope):
		with tf.device(self.device):
			ystack = tf.reshape(classes, [self.batch_size,1, 1, self.num_class])
			embedding = tf.concat(axis=1, values=[embedding, classes])
			h1 = tf.layers.dense(embedding, units=self.dim1, activation=None,
				kernel_initializer=self.initializer, 
				name='dense_1', reuse=scope.reuse)
			h1_relu = tf.nn.relu(self.normalize(h1))
			h1_concat = tf.concat(axis=1, values=[h1_relu, classes])
			h2 = tf.layers.dense(h1_concat, units=self.dim_8*self.dim_8*self.dim2, 
				activation=None, kernel_initializer=self.initializer,
				name='dense_2',	reuse=scope.reuse)
			h2_relu = tf.nn.relu(self.normalize(h2))
			h2_concat = tf.concat(axis=3,
				values=[tf.reshape(h2_relu, shape=[self.batch_size,self.dim_8,self.dim_8,self.dim2]), 
				ystack*tf.ones(shape=[self.batch_size, self.dim_8, self.dim_8, 
				self.num_class])])
			h3 = tf.layers.conv2d_transpose(inputs=h2_concat, filters = 2*self.dim3, 
				kernel_size=[4,4], strides=[2,2], padding='SAME', activation=None,
				kernel_initializer=self.initializer,
				reuse=scope.reuse,name='conv_1')
			h3_relu = tf.nn.relu(self.normalize(h3,flag=True))
            # print(h3.get_shape())
			h3_concat = tf.concat(axis=3,
				values=[tf.reshape(h3_relu, shape=[self.batch_size,self.dim_4,self.dim_4,2*self.dim3]), 
				ystack*tf.ones(shape=[self.batch_size, self.dim_4, self.dim_4, self.num_class])])
			h4 = tf.layers.conv2d_transpose(inputs=h3_concat, filters = 2*self.dim4, 
				kernel_size=[4,4], strides=[2,2], padding='SAME', activation=tf.nn.relu,
				kernel_initializer=self.initializer,
				reuse=scope.reuse,name="conv_2")
			h4_relu = tf.nn.relu(self.normalize(h4,flag=True))
			h4_concat = tf.concat(axis=3,
				values=[tf.reshape(h4_relu, shape=[self.batch_size,self.dim_2,self.dim_2,2*self.dim4]), 
				ystack*tf.ones(shape=[self.batch_size, self.dim_2, self.dim_2, self.num_class])])
			h5 = tf.layers.conv2d_transpose(inputs=h4_concat, filters = 4*self.dim4, 
				kernel_size=[4,4], strides=[2,2], padding='SAME', activation=None,
				kernel_initializer=self.initializer,
				reuse=scope.reuse,name="conv_3")
			h5_relu = tf.nn.relu(self.normalize(h5, flag=True))
			h5_concat = tf.concat(axis=3, 
				values=[h5_relu, ystack*tf.ones(shape=[self.batch_size, self.dim_1, self.dim_1, self.num_class])])
			h6 = tf.layers.conv2d_transpose(inputs=h5_concat, filters = self.dim_channel*self.frames,
				kernel_size=[5,5], strides=[1,1], padding='SAME', activation=None,
				kernel_initializer=self.initializer,
				reuse=scope.reuse, name="conv_4")
			return tf.nn.sigmoid(h6)
	def encoder_image(self, image, scope):
		with tf.device(self.device):
			LeakyReLU = tf.contrib.keras.layers.LeakyReLU(alpha=0.2)
			image_proc = self.normalize(image,flag=True)
			h1 = tf.layers.conv2d(image_proc, filters=48, kernel_size=[4,4],
				strides=[2,2], padding='SAME',
				activation=None,
				kernel_initializer=self.initializer,
				reuse=scope.reuse, name="conv_1")
			h1_relu = self.normalize(LeakyReLU(h1))
			h2 = tf.layers.conv2d(h1_relu, filters=64, kernel_size=[4,4],
				strides=[2,2], padding='SAME',
				activation=None,
				kernel_initializer=self.initializer,
				reuse=scope.reuse, name="conv_2")
			h2_relu = self.normalize(LeakyReLU(h2))
			h3 = tf.layers.conv2d(h2_relu, filters=16, kernel_size=[4,4],
				strides=[2,2], padding='SAME',
				activation=None,
				kernel_initializer=self.initializer,
				reuse=scope.reuse, name="conv_3")
			h3_relu = self.normalize(LeakyReLU(h3))
			h3_reshape = tf.reshape(h3_relu, shape=[self.batch_size, self.dim_8[0]*self.dim_8[1]*16])
			h4 = tf.layers.dense(h3_reshape, units=self.embedding_size+self.num_class_image, 
				activation=None,
				kernel_initializer=self.initializer,
				name='dense_2',
				reuse=scope.reuse)
			return h4 # no activation over last layer of h4
	def decoder_image(self, embedding, zvalue, scope):
		with tf.device(self.device):
			ystack = tf.reshape(zvalue, shape=[self.batch_size, 1,1 , self.zdimension])
			yneed_1 = ystack*tf.ones([self.batch_size, self.dim_4[0], self.dim_4[1], self.zdimension])
			yneed_2 = ystack*tf.ones([self.batch_size, self.dim_2[0], self.dim_2[1], self.zdimension])
			yneed_3 = ystack*tf.ones([self.batch_size, self.dim_8[0], self.dim_8[1], self.zdimension])
			embedding = tf.concat(axis=1, values=[embedding, zvalue])
			h1 = tf.layers.dense(embedding, units=1280, activation=None,
				kernel_initializer=self.initializer, 
				name='dense_1', reuse=scope.reuse)
			h1_relu = tf.nn.relu(self.normalize(h1))
			h1_reshape = tf.reshape(h1_relu, shape=[self.batch_size, self.dim_8[0], self.dim_8[1], 64])
			h1_concat = tf.concat(axis=3, values=[h1_reshape,yneed_3])
			h2 = tf.layers.conv2d_transpose(inputs=h1_concat, filters = 64, 
				kernel_size=[5,5], strides=[2,2], padding='SAME', activation=None,
				kernel_initializer=self.initializer,
				reuse=scope.reuse,name='conv_1')
			h2_relu = tf.nn.relu(self.normalize(h2))
			h2_concat = tf.concat(axis=3, values=[h2_relu, yneed_1])
			h3 = tf.layers.conv2d_transpose(inputs=h2_concat, filters = 32, 
				kernel_size=[5,5], strides=[2,2], padding='SAME', activation=None,
				kernel_initializer=self.initializer,
				reuse=scope.reuse,name='conv_2')
			h3_relu = tf.nn.relu(self.normalize(h3))
			h3_concat = tf.concat(axis=3, values=[h3_relu, yneed_2])
			h4 = tf.layers.conv2d_transpose(inputs=h3_concat, filters = self.dim_channel, 
				kernel_size=[5,5], strides=[2,2], padding='SAME', activation=None,
				kernel_initializer=self.initializer,
				reuse=scope.reuse,name='conv_3')
			return tf.nn.sigmoid(h4)
	def discriminate_image(self, image, zvalue, scope):
		with tf.device(self.device):
			with tf.variable_scope("encoder") as scope:
				embedding = self.encoder_image(image, scope)
			with tf.variable_scope("decoder") as scope:
				image_reconstr = self.encoder_image(embedding, zvalue, scope)
			return tf.sqrt(tf.reduce_mean(tf.square(image - image_reconstr)))
	def build_mode(self):
		with tf.device(self.device):
			embedding = tf.placeholder(tf.float32, [self.batch_size, self.embedding_size])
			classes = tf.placeholder(tf.float32, [self.batch_size,self.num_class])
			r_image = tf.placeholder(tf.float32,[self.batch_size] + self.image_shape)
			real_image = tf.reshape(r_image,[self.batch_size] + self.image_shape)
			with tf.variable_scope("generator") as scope:	
				h4 = self.generate(embedding,classes,scope)
			g_image = h4
			with tf.variable_scope("discriminator") as scope:
				real_value = self.discriminate(real_image,classes,scope)
			with tf.variable_scope("discriminator") as scope:
				scope.reuse_variables()
				fake_value = self.discriminate(g_image,classes,scope)
			d_cost = real_value - fake_value
			g_cost = fake_value
			return embedding, classes, r_image, d_cost, g_cost, fake_value, real_value
