import numpy as np
import tensorflow as tf
import sys
import matplotlib.pyplot as plt
from random import sample

IMG_CHANNELS = 1

def build_convnet(inputs, labels, conv_layer_params, num_hidden_nodes, keep_prob):
    output = inputs
    print output.get_shape()
    
    # Convolutional Layers
    for i, layer in enumerate(conv_layer_params):
        with tf.name_scope('convlayer' + str(i)):
            kernels = tf.Variable(tf.truncated_normal(layer['conv kshape'], stddev=0.1))
            biases = tf.Variable(tf.constant(0.1, shape=layer['bias shape']))
            conv = layer['conv function'](output, kernels, strides=layer['conv stride'], 
                                          padding=layer['conv padding']) + biases
            activation = layer['nonlinearity'](conv)
            output = tf.nn.max_pool(activation, ksize=layer['pooling kshape'],
                                    strides=layer['pooling stride'], 
                                    padding=layer['pooling padding'])
    d1, d2, d3, d4 = [x.value for x in output.get_shape()] 
    print d1, d2, d3, d4

    # Fully Connected Layer
    output = tf.reshape(output, [-1, d2*d3*d4])
    with tf.name_scope('hidden_layer'):
        weights = tf.Variable(tf.truncated_normal([d2*d3*d4, num_hidden_nodes], stddev=0.1))
        biases = tf.Variable(tf.constant(0.1, shape=[num_hidden_nodes]))
        linear = tf.matmul(output, weights) + biases
        output = tf.nn.relu(linear) #tf.nn.sigmoid(linear)
        output = tf.nn.dropout(output, keep_prob)

    # Readout Layer
    with tf.name_scope('readout_layer'):
        weights = tf.Variable(tf.truncated_normal([num_hidden_nodes, 2], stddev=0.1))
        biases = tf.Variable(tf.constant(0.1, shape=[2]))
        logits = tf.matmul(output, weights) + biases
        probs = tf.nn.softmax(logits)

    # Loss layer & training operation
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, labels))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

    return probs, loss, train_step

def evaluation(logits, labels):
    correct = tf.equal(tf.argmax(logits,1), tf.argmax(labels, 1))
    return tf.reduce_mean(tf.cast(correct, tf.int32))

def make_batches(input, batch_size):
    return [np.array(input[i:i+batch_size]) for i in xrange(0, len(input), batch_size)]

def train_nn(train_input, train_labels, val_input, val_labels, 
             conv_layer_params, num_hidden_nodes, num_epochs, batch_size):
    img_sz = train_input[0].shape[0]
    #batch_size = find_even_batch_size(train_input, min_batch_size)
    print "Batch Size: {}".format(batch_size)
    batched_train_input = make_batches(train_input, batch_size)
    print "Num Batches: {}".format(len(batched_train_input))
    batched_train_labels = make_batches(train_labels, batch_size)
    batched_val_input = make_batches(val_input, batch_size)
    batched_val_labels = make_batches(val_labels, batch_size)
    input_placeholder = tf.placeholder(tf.float32, shape=(batch_size, img_sz, img_sz, IMG_CHANNELS))
    labels_placeholder = tf.placeholder(tf.float32, shape=(batch_size, 2))
    keep_prob = tf.placeholder(tf.float32)
    probs, loss, train_op = build_convnet(input_placeholder, labels_placeholder, conv_layer_params,
                                          num_hidden_nodes, keep_prob)
 
    saver = tf.train.Saver(max_to_keep=1)
    with tf.Session() as sess:
        init = tf.initialize_all_variables()
        sess.run(init)
        
        train_loss_per_epoch = []
        val_loss_per_epoch = []    
        best_loss = np.inf
        best_epoch = 0
        for epoch in xrange(num_epochs):
            print "Training Epoch: {}".format(epoch)

            batch_losses_train = []
            for batch in range(len(batched_train_input)): #get rid of sample 
                print batch,
                feed_dict = {input_placeholder: batched_train_input[batch], 
                             labels_placeholder: batched_train_labels[batch],
                             keep_prob: 0.8}
                _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
                batch_losses_train.append(loss_value)
            train_loss_per_epoch.append(np.mean(batch_losses_train))
                        
            batch_losses_val = []
            for batch in range(len(batched_val_input)): #get rid of sample
                feed_dict = {input_placeholder: batched_val_input[batch], 
                             labels_placeholder: batched_val_labels[batch],
                             keep_prob: 1}
                val_loss = sess.run(loss, feed_dict=feed_dict)
                batch_losses_val.append(val_loss)
            val_loss_per_epoch.append(np.mean(batch_losses_val))
                
            print "\nTraining Loss: {}  Validation Loss: {}".format(train_loss_per_epoch[epoch], 
                                                                    val_loss_per_epoch[epoch])
            if val_loss_per_epoch[epoch] < best_loss:
                best_loss = val_loss_per_epoch[epoch]
                best_epoch = epoch
                saver.save(sess,  "nn_best.ckpt")
        saver.restore(sess, "nn_best.ckpt")
        
        # plot losses
        fig, ax = plt.subplots(2, sharex=True)
        ax[0].plot(range(len(train_loss_per_epoch)), train_loss_per_epoch, label="training loss",c='b')
        ax[1].plot(range(len(val_loss_per_epoch)), val_loss_per_epoch, label="validation loss", c='g')
        plt.legend()
        plt.xlabel("epoch")
        plt.ylabel("mean cross-entropy loss")
        plt.axvline(best_epoch, linewidth=2, color='r')
        plt.savefig("loss_curves_nn.png")
        
        def nn(inputs):
            session = tf.Session()
            saver.restore(session, "nn_best.ckpt")
            batched_inputs = make_batches(inputs, batch_size)
            all_probs = []
            for batch in xrange(len(batched_inputs)):
                fd = {input_placeholder: batched_inputs[batch], 
                      labels_placeholder: np.zeros((batch_size, 2)), 
                      keep_prob: 1}
                prob_vals = session.run(probs, feed_dict=fd)
                all_probs.extend([prob_vals[i,:] for i in range(prob_vals.shape[0])])
            return all_probs

        return nn
