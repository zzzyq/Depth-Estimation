import tensorflow as tf 

# 2D convolution layer structure
def conv2d(input, name, filter_size, strides_size, padding_model, reuse):
	with tf.variable_scope(name) as scope:
		if reuse is True:
			scope.reuse_variables()
		filter = tf.get_variable('filter', filter_size, initializer = tf.truncated_normal_initializer(mean = 0, stddev = 0.01), trainable = True)
		filter = variable_with_weight_decay(
			'filter',
			filter_size,
			stddev=0.01,
			wd=0.01,
		)
		# do convolution
		conv = tf.nn.conv2d(input, filter, strides = strides_size, padding = padding_model)
		biases = tf.get_variable('biases', filter_size[3], initializer = tf.constant_initializer(0.1))
		# add biases to conv
		output = tf.nn.bias_add(conv, biases)
		output = tf.nn.relu(output, name = name)
		return output

# max pooling function
def pooling(input, name, ksize, strides_size, padding_model):
	output = tf.nn.max_pool(input, ksize, strides_size, padding_model, name = name)
	return output

# fully connected layer structure
def fc(input, name, shape, reuse):
	with tf.variable_scope(name) as scope:
		if reuse is True:
			scope.reuse_variables()
		# -1 is inference shape
		flat = tf.reshape(input, [-1, shape[0]])
		weights = variable_with_weight_decay(
			'weights',
		 	shape,
		 	stddev=0.01,
		 	wd=0.04,
	 	)
		biases = tf.get_variable('biases', shape[1], initializer = tf.constant_initializer(0.1))
		output = tf.nn.relu_layer(flat, weights, biases, name=name)
		return output

# get layer's variable with weight decay
def variable_with_weight_decay(name, shape, stddev, wd):
	with tf.variable_scope(name) as scope:
		var = tf.get_variable(name, shape, initializer = tf.truncated_normal_initializer(mean = 0, stddev = 0.01), trainable = True)
		if wd is not None:
			weight_loss = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
			tf.add_to_collection('losses', weight_loss)
			return var

