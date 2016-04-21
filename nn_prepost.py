import numpy as np
import matplotlib.pyplot as plt
import sys
import pickle
from sklearn.decomposition import PCA
from skimage.transform import downscale_local_mean
from sklearn.cluster import DBSCAN
import itertools
import os.path
from matplotlib.patches import Rectangle
from load import *
from test import *
from nn_arch import *
from nn import train_nn
from random import sample, shuffle

IMG_CHANNELS = 1

def load_data():
    # generate filenames
    files = []
    for i in range(1,4):
        for j in range(1,5):
            prefix = 'data/AMG%d_exp%d'%(i,j)
            files.append( (prefix+'.tif', prefix+'.zip') )
        
    # load data
    data = []
    for i,(s,r) in enumerate(files):
        if i==8: # lolwtf
            data.append((load_stack(s), load_rois(r, 512, 512, xdisp=9, ydisp=-1)))
        else:
            data.append((load_stack(s), load_rois(r, 512, 512)))
    return data

def get_centroids(rois, radius):
    new_rois = np.zeros(rois.shape)
    for i,r in enumerate(rois):
        x,y = np.where(r!=0)
        x,y = int(x.mean()), int(y.mean())
        new_rois[i, x-radius:x+radius, y-radius:y+radius] = 1
    return new_rois

def preprocess(data, radius):
    for i, (stk,roi) in enumerate(data):
        # Normalize
        stk = np.divide(stk, stk.mean())
        # Downscale
        stk = downscale_local_mean(stk, (1,2,2))
        roi = downscale_local_mean(roi, (1,2,2))
        new_stk = np.zeros((stk.shape[1], stk.shape[2], IMG_CHANNELS))
        # Uncomment following 2 lines to use more channels
        #new_stk[:,:,2] = stk.max(axis=0)
        #new_stk[:,:,1] = np.std(stk, axis=0)
        new_stk[:,:,0] = np.mean(stk, axis=0)
        roi_centroids = get_centroids(roi, radius).max(axis=0)
        data[i] = (new_stk, roi_centroids, roi.max(axis=0))
    return data
       
def clips_and_labels_stk(data, sz, step):
    stk, roi_centroids, _ = data
    rows, cols, channels = stk.shape
    clips = []
    labels = []
    for ulr, ulc in itertools.product(range(0,rows-sz,step), range(0,cols-sz,step)):
        clip = stk[ulr : ulr+sz, ulc : ulc+sz, :]
        if roi_centroids[ulr + int(sz/2), ulc + int(sz/2)] == 1:
            labels.append(np.array([0,1]))
        else:
            labels.append(np.array([1,0]))
        clips.append(clip)
    return (clips, labels)
 
def rotate_augment(clips, labels):
    new_clips = []
    new_labels = []
    for clip, label in zip(clips, labels):
        c90 = np.rot90(clip,k=1)
        c180 = np.rot90(clip, k=2)
        c270 = np.rot90(clip, k=3)
        new_clips.extend([clip, c90, c180, c270])
        new_labels.extend([label, label, label, label])
    return new_clips, new_labels

def labels_to_stk(labels, orig_shape, sz, step):
    rows, cols = orig_shape
    stk = np.zeros((rows, cols))
    i = 0
    for r,c in itertools.product(range(int(sz/2), int(rows-sz/2), step), 
                                 range(int(sz/2), int(cols-sz/2), step)):
        assert(len(labels[i].shape) == 1 and (labels[i].shape)[0] == 2)
        stk[r,c] = np.argmax(labels[i])
        i += 1
    return stk

def equalize_posneg(clips, labels):
    both = zip(clips, labels)
    negs = filter(lambda x: x[1][0] == 1, both)
    poss = filter(lambda x: x[1][1] == 1, both)
    num_pos = len(poss)
    num_neg = len(negs)
    print "{} negative examples, {} positive examples before equalizing".format(num_neg, num_pos)
    downsampled_negs = sample(negs, num_pos)
    new_clips_labels = []
    new_clips_labels.extend(poss)
    new_clips_labels.extend(downsampled_negs)
    shuffle(new_clips_labels)
    new_clips, new_labels = zip(*new_clips_labels)
    return new_clips, new_labels
    

def main():
    clip_sz = int(sys.argv[1])
    clip_step = int(sys.argv[2])
    min_batch_size = int(sys.argv[3])
    num_epochs = int(sys.argv[4])
    load = True if (len(sys.argv) > 5 and sys.argv[5] == 'load') else False
    num_hidden_nodes = 1024
    radius = 2
    net = mnist_net_3x3

    data = preprocess(load_data(), radius)
    train_data = data[0:8]
    val_data = data[8:10]
    test_data = data[10:12]
       
    # clips: list of 3d arrays, labels: list of 2-element vectors
    train_clips_labels = [clips_and_labels_stk(t, clip_sz, clip_step) for t in train_data] 
    val_clips_labels = [clips_and_labels_stk(t, clip_sz, clip_step) for t in val_data]
    test_clips_labels = [clips_and_labels_stk(t, clip_sz, clip_step) for t in test_data]    
   
    # rotate clips to get even more training data
    train_clips_labels_rot = [rotate_augment(tc, tl) for (tc, tl) in train_clips_labels]
    val_clips_labels_rot = [rotate_augment(vc, vl) for (vc, vl) in val_clips_labels]
   
    # concatenate all training and validation examples 
    train_clips_rot_all = [c for t in train_clips_labels_rot for c in t[0]]
    train_labels_rot_all = [c for t in train_clips_labels_rot for c in t[1]]
    val_clips_rot_all = [c for t in val_clips_labels_rot for c in t[0]]
    val_labels_rot_all = [c for t in val_clips_labels_rot for c in t[1]]

    # throw out some negative examples to help equalize pos/neg ratio
    train_clips_rot_all, train_labels_rot_all = equalize_posneg(train_clips_rot_all, train_labels_rot_all)
    val_clips_rot_all, val_labels_rot_all = equalize_posneg(val_clips_rot_all, val_labels_rot_all)
         
    # find a batch size
    batch_size = min_batch_size
    while len(train_clips_labels[0][0]) % batch_size != 0:
        batch_size += 1

    # ensure batch_size divides evenly into training clips
    train_clips_rot_all = train_clips_rot_all[0:-(len(train_clips_rot_all)%batch_size)]
    train_labels_rot_all = train_labels_rot_all[0:-(len(train_labels_rot_all)%batch_size)]
    val_clips_rot_all = val_clips_rot_all[0:-(len(val_clips_rot_all)%batch_size)]
    val_labels_rot_all = val_labels_rot_all[0:-(len(val_labels_rot_all)%batch_size)]

    print "Number of training frames: {}".format(len(train_clips_rot_all))
    
    if not load:
        # train neural net
        nn = train_nn(train_clips_rot_all, train_labels_rot_all, val_clips_rot_all, val_labels_rot_all, 
                      net, num_hidden_nodes, num_epochs, batch_size)
    
        # classify data
        train_nn_labels = [nn(t[0]) for t in train_clips_labels] 
        val_nn_labels = [nn(t[0]) for t in val_clips_labels] 
        test_nn_labels = [nn(t[0]) for t in test_clips_labels]
        
        pickle.dump(train_nn_labels, open("nn_train_labels.pickle",'wb'))
        pickle.dump(val_nn_labels, open("nn_val_labels.pickle", 'wb'))
        pickle.dump(test_nn_labels, open("nn_test_labels.pickle", 'wb'))
    
    else:
        train_nn_labels = pickle.load(open("nn_train_labels.pickle",'rb'))
        val_nn_labels = pickle.load(open("nn_val_labels.pickle", 'rb'))
        test_nn_labels = pickle.load(open("nn_test_labels.pickle", 'rb'))
    
    
    # plot things
    plt.figure()
    plt.imshow(train_data[0][1], cmap="Greys")
    plt.savefig("nn_train_actual.png")
    plt.figure()
    plt.imshow(labels_to_stk(train_clips_labels[0][1], train_data[0][1].shape, 
                             clip_sz, clip_step), cmap="Greys")
    plt.figure()
    plt.imshow(labels_to_stk(train_nn_labels[0], train_data[0][1].shape,
                             clip_sz, clip_step), cmap="Greys")
    plt.savefig("nn_train_pred.png")
    

    plt.figure()
    plt.imshow(val_data[0][1], cmap="Greys")
    plt.savefig("nn_val_actual.png")
    plt.figure()
    plt.imshow(labels_to_stk(val_clips_labels[0][1], val_data[0][1].shape, 
                             clip_sz, clip_step), cmap="Greys")
    plt.figure()
    plt.imshow(labels_to_stk(val_nn_labels[0], val_data[0][1].shape,
                             clip_sz, clip_step), cmap="Greys")
    plt.savefig("nn_val_pred.png")
    plt.show()
    

if __name__ == "__main__":
    main()
