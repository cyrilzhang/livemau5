import tensorflow as tf 

IMG_CHANNELS = 1

onelayer_7x32 = [
    {'conv function': tf.nn.conv2d,
     'conv kshape':  [7,7,IMG_CHANNELS,32],
     'bias shape':   [32],
     'conv stride':  [1,1,1,1],
     'conv padding': 'SAME',
     'nonlinearity': tf.nn.relu,
     'pooling kshape': [1,2,2,1],
     'pooling stride': [1,2,2,1],
     'pooling padding': 'SAME'}
]


twolayer_5x32 = [
    {'conv function': tf.nn.conv2d,
     'conv kshape':  [5,5,IMG_CHANNELS,32],
     'bias shape':   [32],
     'conv stride':  [1,1,1,1],
     'conv padding': 'SAME',
     'nonlinearity': tf.nn.relu,
     'pooling kshape': [1,2,2,1],
     'pooling stride': [1,2,2,1],
     'pooling padding': 'SAME'},

    {'conv function': tf.nn.conv2d,
     'conv kshape':  [5,5,32,64],
     'bias shape':   [64],
     'conv stride':  [1,1,1,1],
     'conv padding': 'SAME',
     'nonlinearity': tf.nn.relu,
     'pooling kshape': [1,2,2,1],
     'pooling stride': [1,2,2,1],
     'pooling padding': 'SAME'}
]

twolayer_5x64 = [
    {'conv function': tf.nn.conv2d,
     'conv kshape':  [5,5,IMG_CHANNELS,64],
     'bias shape':   [64],
     'conv stride':  [1,1,1,1],
     'conv padding': 'SAME',
     'nonlinearity': tf.nn.relu,
     'pooling kshape': [1,2,2,1],
     'pooling stride': [1,2,2,1],
     'pooling padding': 'SAME'},

    {'conv function': tf.nn.conv2d,
     'conv kshape':  [5,5,64,128],
     'bias shape':   [128],
     'conv stride':  [1,1,1,1],
     'conv padding': 'SAME',
     'nonlinearity': tf.nn.relu,
     'pooling kshape': [1,2,2,1],
     'pooling stride': [1,2,2,1],
     'pooling padding': 'SAME'}
]

fivelayer_3x24 = [
    {'conv function': tf.nn.conv2d,
     'conv kshape':  [3,3,IMG_CHANNELS,24],
     'bias shape':   [24],
     'conv stride':  [1,1,1,1],
     'conv padding': 'SAME',
     'nonlinearity': tf.nn.relu,
     'pooling kshape': [1,1,1,1],
     'pooling stride': [1,1,1,1],
     'pooling padding': 'SAME'},

    {'conv function': tf.nn.conv2d,
     'conv kshape':  [3,3,24,48],
     'bias shape':   [48],
     'conv stride':  [1,1,1,1],
     'conv padding': 'SAME',
     'nonlinearity': tf.nn.relu,
     'pooling kshape': [1,2,2,1],
     'pooling stride': [1,2,2,1],
     'pooling padding': 'SAME'},

    {'conv function': tf.nn.conv2d,
     'conv kshape':  [3,3,48,72],
     'bias shape':   [72],
     'conv stride':  [1,1,1,1],
     'conv padding': 'SAME',
     'nonlinearity': tf.nn.relu,
     'pooling kshape': [1,1,1,1],
     'pooling stride': [1,1,1,1],
     'pooling padding': 'SAME'},

    {'conv function': tf.nn.conv2d,
     'conv kshape':  [3,3,72,96],
     'bias shape':   [96],
     'conv stride':  [1,1,1,1],
     'conv padding': 'SAME',
     'nonlinearity': tf.nn.relu,
     'pooling kshape': [1,2,2,1],
     'pooling stride': [1,2,2,1],
     'pooling padding': 'SAME'},

    {'conv function': tf.nn.conv2d,
     'conv kshape':  [3,3,96,120],
     'bias shape':   [120],
     'conv stride':  [1,1,1,1],
     'conv padding': 'SAME',
     'nonlinearity': tf.nn.relu,
     'pooling kshape': [1,1,1,1],
     'pooling stride': [1,1,1,1],
     'pooling padding': 'SAME'}
]


threelayer_753x32 = [
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
     'conv kshape':  [5,5,32,64],
     'bias shape':   [64],
     'conv stride':  [1,1,1,1],
     'conv padding': 'SAME',
     'nonlinearity': tf.nn.relu,
     'pooling kshape': [1,2,2,1],
     'pooling stride': [1,2,2,1],
     'pooling padding': 'SAME'},

    {'conv function': tf.nn.conv2d,
     'conv kshape':  [3,3,64,96],
     'bias shape':   [96],
     'conv stride':  [1,1,1,1],
     'conv padding': 'SAME',
     'nonlinearity': tf.nn.relu,
     'pooling kshape': [1,1,1,1],
     'pooling stride': [1,1,1,1],
     'pooling padding': 'SAME'}
]





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
   
