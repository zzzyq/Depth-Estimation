import tensorflow as tf
import math
from layers import conv2d 
from layers import pooling
from layers import fc

def coarseNet(image, reuse):
	# coarse 1
	coarse_1_conv = conv2d(image, 'coarse_1_conv', [11, 11, 3, 96], [1, 4, 4, 1], 'VALID', reuse=reuse)
	coarse_1 = pooling(coarse_1_conv, 'coarse_1_pooling', [1, 3, 3, 1], [1, 2, 2, 1], 'VALID')
	# coarse 2
	coarse_2_conv = conv2d(coarse_1, 'coarse_2_conv', [5, 5, 96, 256], [1, 1, 1, 1], 'VALID', reuse=reuse)
	coarse_2 = pooling(coarse_2_conv, 'coarse_2_pooling', [1, 3, 3, 1], [1, 2, 2, 1], 'SAME')
	# coarse 3
	coarse_3 = conv2d(coarse_2, 'coarse_3_conv', [3, 3, 256, 384], [1, 1, 1, 1], 'VALID', reuse=reuse)
	# coarse 4
	coarse_4 = conv2d(coarse_3, 'coarse_4_conv', [3, 3, 384, 384], [1, 1, 1, 1], 'VALID', reuse=reuse)
	# coarse 5
	coarse_5 = conv2d(coarse_4, 'coarse_5_conv', [3, 3, 384, 256], [1, 1, 1, 1], 'VALID', reuse=reuse)
	# coarse 6
	coarse_6 = fc(coarse_5, 'coarse_6_fc', [10*6*256, 4096], reuse=reuse)
	# coarse 7 reshape to image size
	coarse_7_fc = fc(coarse_6, 'coarse_7_fc', [4096, 4070], reuse=reuse)
	coarse_7 = tf.reshape(coarse_7_fc, [-1, 55, 74, 1])
	return coarse_7


def fineNet(image, coarse_output, reuse):
	# fine_1
	fine_1_conv = conv2d(image, 'fine_1_conv', [9, 9, 3, 63], [1, 2, 2, 1], 'VALID', reuse=reuse)
	fine_1 = pooling(fine_1_conv, 'fine_1_pooling', [1, 3, 3, 1], [1, 2, 2, 1], 'SAME')
	# concatenate layer
	fine_2 = tf.concat([fine_1, coarse_output], 3)
	# fine_3
	fine_3 = conv2d(fine_2, 'fine_3_conv', [5, 5, 64, 64], [1, 1, 1, 1], 'SAME', reuse=reuse)
	# fine_4
	fine_4 = conv2d(fine_3, 'fin3_4_conv', [5, 5, 64, 1], [1, 1, 1, 1], 'SAME', reuse=reuse)
	return fine_4


def loss(predict, truth):
	predict = tf.reshape(predict, [-1, 55*74])
	truth = tf.reshape(truth, [-1, 55*74])
	diff = tf.subtract(predict, truth)
	diff_square = tf.square(diff)
	sum_diff_square = tf.reduce_sum(diff_square, 1)
	sum_diff = tf.reduce_sum(diff)
	square_sum_diff = tf.square(sum_diff)
	loss = tf.reduce_mean(sum_diff_square / 55.0*74.0 - 0.5 * square_sum_diff / math.pow(55*74, 2))
	tf.add_to_collection('losses', loss)
	return tf.add_n(tf.get_collection('losses'), name='total_loss')

def test(path):
	testimage = tf.read_file(path)
	testimage = tf.image.decode_jpeg(testimage, channels=3)
	testimage = tf.image.resize_images(testimage, (228, 304))
	testimage = tf.reshape(testimage, [1, 228, 304, 3]);
	test_coarse = coarseNet(testimage, reuse = True)
	test_fine = fineNet(testimage, test_coarse, reuse = True)
	test_out = tf.reshape(test_fine, [55, 74, 1])
	return test_out

