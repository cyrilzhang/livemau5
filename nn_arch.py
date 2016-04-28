import tensorflow as tf 

IMG_CHANNELS = 1

mnist_net_7x7 = [
    {'conv function': tf.nn.conv2d,
     'conv kshape':  [7,7,IMG_CHANNELS,32],
     'bias shape':   [32],
     'conv stride':  [1,1,1,1],
     'conv padding': 'SAME',
     'nonlinearity': tf.nn.relu,
     'pooling kshape': [1,2,2,1],
     'pooling stride': [1,2,2,1],
     'pooling padding': 'SAME'},

    {'conv function': tf.nn.conv2d,
     'conv kshape':  [7,7,32,64],
     'bias shape':   [64],
     'conv stride':  [1,1,1,1],
     'conv padding': 'SAME',
     'nonlinearity': tf.nn.relu,
     'pooling kshape': [1,2,2,1],
     'pooling stride': [1,2,2,1],
     'pooling padding': 'SAME'}
    ]

mnist_net_3x3 = [
    {'conv function': tf.nn.conv2d,
     'conv kshape':  [3,3,IMG_CHANNELS,32],
     'bias shape':   [32],
     'conv stride':  [1,1,1,1],
     'conv padding': 'SAME',
     'nonlinearity': tf.nn.relu,
     'pooling kshape': [1,2,2,1],
     'pooling stride': [1,2,2,1],
     'pooling padding': 'SAME'},

    {'conv function': tf.nn.conv2d,
     'conv kshape':  [3,3,32,64],
     'bias shape':   [64],
     'conv stride':  [1,1,1,1],
     'conv padding': 'SAME',
     'nonlinearity': tf.nn.relu,
     'pooling kshape': [1,2,2,1],
     'pooling stride': [1,2,2,1],
     'pooling padding': 'SAME'}
    ]
   
