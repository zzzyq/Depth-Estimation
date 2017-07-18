import tensorflow as tf 
from tensorflow.python.platform import gfile
import numpy as np
from PIL import Image

input_height = 228
input_width = 304
output_height = 55
output_width = 74

def Input(csv_path, batch_size):
	filename_queue = tf.train.string_input_producer([csv_path], shuffle = False)
	reader = tf.TextLineReader()
	key, value = reader.read(filename_queue)
	record_defaults = [["image"], ["depth"]]
	image_name, depth_name = tf.decode_csv(value, record_defaults)
	# image
	jpg = tf.read_file(image_name)
	image = tf.image.decode_jpeg(jpg, channels = 3)
	image = tf.cast(image, tf.float32)  
	# depth
	png = tf.read_file(depth_name)
	depth = tf.image.decode_png(png, channels = 1)
	depth = tf.cast(depth, tf.float32)
	#depth = tf.div(depth, [255.0])
	# resize
	image = tf.image.resize_images(image, (input_height, input_width))
	depth = tf.image.resize_images(depth, (output_height, output_width))
	# build batch
	images, depths = tf.train.batch(
		[image, depth],
		batch_size=batch_size,
		num_threads=4,
		capacity= 50 + 2 * batch_size,
		)
	return images, depths

def Output(images, depths, predicts, dir):
	if not gfile.Exists(dir):
		gfile.MakeDirs(dir)
	for i, (image, depth, predict) in enumerate(zip(images, depths, predicts)):
		# save original images
		image = Image.fromarray(np.uint8(image))
		image_name = "%s/%05d_org.png" % (dir, i)
		image.save(image_name)
		# save depth
		depth = depth.transpose(2, 0, 1)
		depth = Image.fromarray(np.uint8(depth[0]), mode="L")
		depth_name = "%s/%05d_truth.png" % (dir, i)
		depth.save(depth_name)
		# save predict
		predict = predict.transpose(2, 0, 1)
		if np.max(predict) != 0:
			predict = (predict / np.max(predict)) * 255.0
		else:
			predict = predict * 255.0
		predict = Image.fromarray(np.uint8(predict[0]), mode="L")
		predict_name = "%s/%05d_pred.png" % (dir, i)
		predict.save(predict_name)

def singleOutput(predict, dir, step):
	if not gfile.Exists(dir):
		gfile.MakeDirs(dir)
	predict = predict.transpose(2, 0, 1)
	if np.max(predict) != 0:
		predict = (predict / np.max(predict)) * 255.0
	else:
		predict = predict * 255.0
	predict = Image.fromarray(np.uint8(predict[0]), mode="L")
	predict_name = "%s/%05d_pred.png" % (dir, step)
	predict.save(predict_name)

