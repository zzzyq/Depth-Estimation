import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt;  
import model
from IO import Input
from IO import Output
from IO import singleOutput

MAX_STEPS = 100 # epoch num
LOG_DEVICE_PLACEMENT = False # if true, GPU is used
BATCH_SIZE = 16 # batch size for training 
TRAIN_FILE = "train.csv" # training dataset
TEST_FILE1 = "test_data/01399.jpg"
TEST_FILE2 = "test_data/01249.jpg"

graph = tf.Graph()
with graph.as_default():
	# read training data from csv file and set batch size
	images, depths = Input("train2.csv", 8)
	# coarse network
	out_coarse = model.coarseNet(images, reuse = False)
	# fine network
	out_fine = model.fineNet(images, out_coarse, reuse = False)
	loss = model.loss(out_fine, depths)
	#imgout = tf.cast(out, tf.uint8)
	#imgout = tf.image.encode_jpeg(imgout, "grayscale")
	coarse_var_1_5 = []
	coarse_var_6_7 = []
	fine_var = []
	for variable in tf.all_variables():
		variable_name = variable.name 
		if variable_name.find('coarse_1_conv') >= 0:  
			coarse_var_1_5.append(variable)
		if variable_name.find('coarse_2_conv') >= 0:  
			coarse_var_1_5.append(variable)
		if variable_name.find('coarse_3_conv') >= 0:  
			coarse_var_1_5.append(variable)
		if variable_name.find('coarse_4_conv') >= 0:  
			coarse_var_1_5.append(variable)
		if variable_name.find('coarse_5_conv') >= 0:  
			coarse_var_1_5.append(variable)
		if variable_name.find('coarse_6_fc') >= 0:  
			coarse_var_6_7.append(variable)
		if variable_name.find('coarse_7_fc') >= 0:  
			coarse_var_6_7.append(variable)
		if variable_name.find('fine_1_conv') >= 0:  
			fine_var.append(variable)
		if variable_name.find('fine_3_conv') >= 0:  
			fine_var.append(variable)
		if variable_name.find('fine_4_conv') >= 0:  
			fine_var.append(variable)
	train_op1 = tf.train.AdamOptimizer(1e-4).minimize(loss, var_list = coarse_var_1_5)
	train_op2 = tf.train.AdamOptimizer(1e-2).minimize(loss, var_list = coarse_var_6_7)
	train_op3 = tf.train.AdamOptimizer(1e-4).minimize(loss, var_list = fine_var)
	train_op = tf.group(train_op1, train_op2, train_op3)
	# train_op2 = tf.train.AdamOptimizer(1e-4).minimize(loss, var_list = var)
	# train_op1 = tf.train.AdamOptimizer(1e-4).minimize(loss, var_list = var)

	# w = tf.write_file("t.png", imgout)
	#
	# testimage = tf.read_file("test_data/01399.jpg")
	# testimage = tf.image.decode_jpeg(testimage, channels=3)
	# testimage = tf.image.resize_images(testimage, (228, 304))
	# testimage = tf.reshape(testimage, [1, 228, 304, 3]);
	# test_coarse = model.coarseNet(testimage, reuse = True)
	# test_fine = model.fineNet(testimage, test_coarse, reuse = True)
	test_out_1 = model.test(TEST_FILE1)
	test_out_2 = model.test(TEST_FILE2)

	sess = tf.Session(config=tf.ConfigProto(log_device_placement=LOG_DEVICE_PLACEMENT)) 
	sess.run(tf.global_variables_initializer())
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(sess=sess, coord=coord)

	for step in range(MAX_STEPS):
		sess.run(train_op)
		if step % 10 == 0 :
			images_val, depths_val, predict_val = sess.run([images, depths, out_fine])
			test_val = sess.run(test_out_1)
			test_val1 = sess.run(test_out_2)
			#print test_val.shape
			print sess.run(loss) + step
			Output(images_val, depths_val, predict_val, "results/step_%05d" % step)
			singleOutput(test_val1, "test_results/test2", step)
			singleOutput(test_val, "test_results/test1", step)
	#print sess.run(loss)
	# sess.run(train_step)
	# if i % 10 == 0:
	# 	print sess.run(loss)
	# 	sess.run(w)