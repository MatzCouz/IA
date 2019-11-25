import tensorflow as tf
import numpy as np
from collections import deque

ACTIONS = 3
GAMMA = 0.99
OBSERVE = 1000
EXPLORE = 20000
FINAL_EPSILON = 0.001
INITIAL_EPSILON = 0.99
REPLAY_MEMORY_SIZE = 50000
BATCH = 32
FRAME_PER_ACTION = 1

class Model:
	def __init__(self):
		def conv_layer(x, conv, stride = 1):
			return tf.nn.conv2d(x, conv, [1, stride, stride, 1], padding = 'SAME')
		
		def pooling(x, k = 2, stride = 2):
			return tf.nn.max_pool(x, ksize = [1, k, k, 1], strides = [1, stride, stride, 1], padding = 'SAME')

		self.middle_game = False
		self.memory = deque()
		self.initial_stack_images = np.zeros((80, 80, 4))
		self.X = tf.placeholder("float", [None, 80, 80, 4])
		self.action_space = tf.placeholder("float", [None, 2])
		self.action_left = tf.placeholder("float", [None, 2])
		self.action_right = tf.placeholder("float", [None, 2])
		self.Y = tf.placeholder("float", [None])