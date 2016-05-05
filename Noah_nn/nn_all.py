import numpy as np
import matplotlib.pyplot as plt
import sys
import cPickle as pickle
from skimage.transform import downscale_local_mean
import itertools
import os.path
from matplotlib.patches import Rectangle
from load import *
from test import *
from nn_arch import *
from nn import train_nn
from random import sample, shuffle
from skimage import measure
from scipy.ndimage import zoom

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
       
def clips_and_labels_stk(data, sz, step):
    stk, roi_centroids = data
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
    max_rows = int(rows-sz/2) if sz%2 == 0 else int(rows-sz/2)-1
    max_cols = int(cols-sz/2) if sz%2 == 0 else int(cols-sz/2)-1
    for r,c in itertools.product(range(int(sz/2), max_rows, step), 
                                 range(int(sz/2), max_cols, step)):
        assert(len(labels[i].shape) == 1 and (labels[i].shape)[0] == 2)
        stk[r,c] = labels[i][1] - labels[i][0]
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

def improve_contrast(data):
    new_data = []
    for i, (stk,roi) in enumerate(data):
        low_p = np.percentile(stk.flatten(), 3)
        high_p = np.percentile(stk.flatten(), 99)
        new_stk = np.clip(stk, low_p, high_p)
        new_stk = new_stk - new_stk.mean()
        new_stk = np.divide(new_stk-np.min(new_stk), np.max(new_stk) - np.min(new_stk))
        new_data.append((new_stk, roi))
    return new_data

def downscale(data, downscale_factor):
    new_data = []
    for i, (stk,roi) in enumerate(data):
        new_stk = downscale_local_mean(stk, (1,downscale_factor,downscale_factor))
        new_roi = downscale_local_mean(roi, (1,downscale_factor,downscale_factor))
        new_data.append((new_stk, new_roi))
    return new_data

def flatten(data, flatten_fn):
    new_data = []
    for i, (stk,roi) in enumerate(data):
        new_stk = np.zeros((stk.shape[1], stk.shape[2], 1))
        new_stk[:,:,0] = flatten_fn(stk)
        new_roi = roi.max(axis=0)
        new_data.append((new_stk, new_roi))
    return new_data


def prepare(data, flatten_fn, min_batch_size=100, downscale_factor=2, 
            clip_sz=15, clip_step=1, directory='.'):
    # preprocess
    data_ic = improve_contrast(data)
    data_downscale = downscale(data_ic, downscale_factor)
    data_flatten = flatten(data_downscale, flatten_fn)
    pickle.dump(data, open(directory+"/pickles/data_raw.pickle", 'wb'), protocol=2)
    pickle.dump(data_ic, open(directory+"/pickles/data_ic.pickle", 'wb'), protocol=2)
    pickle.dump(data_downscale, open(directory+"/pickles/data_downscale.pickle", 'wb'), protocol=2)
    pickle.dump(data_flatten, open(directory+"/pickles/data_flatten.pickle", 'wb'), protocol=2)

    # sliding window
    data_clips_labels = [clips_and_labels_stk(s, clip_sz, clip_step) for s in data_flatten]
    pickle.dump(data_clips_labels, open(directory+"/pickles/data_clips_labels.pickle", 'wb'), protocol=2)

    # find a batch size
    batch_size = min_batch_size
    while len(data_clips_labels[0][0]) % batch_size != 0:
        batch_size += 1
    pickle.dump(batch_size, open(directory+"/pickles/batch_size.pickle", 'wb'), protocol=2)

    # prep training data
    train_clips_labels = data_clips_labels[0:8]
    train_clips_labels_rot = [rotate_augment(tc, tl) for (tc, tl) in train_clips_labels]
    train_clips_rot_all = [c for t in train_clips_labels_rot for c in t[0]]
    train_labels_rot_all = [c for t in train_clips_labels_rot for c in t[1]]
    train_clips_rot_all, train_labels_rot_all = equalize_posneg(train_clips_rot_all, train_labels_rot_all)
    train_clips_rot_all = train_clips_rot_all[0:-(len(train_clips_rot_all)%batch_size)]
    train_labels_rot_all = train_labels_rot_all[0:-(len(train_labels_rot_all)%batch_size)]
    pickle.dump(train_clips_rot_all, open(directory+"/pickles/train_clips_rot_all.pickle", 'wb'), protocol=2)
    pickle.dump(train_labels_rot_all, open(directory+"/pickles/train_labels_rot_all.pickle", 'wb'), protocol=2)

    # prep validation data
    val_clips_labels = data_clips_labels[8:10]
    val_clips_labels_rot = [rotate_augment(tc, tl) for (tc, tl) in val_clips_labels]
    val_clips_rot_all = [c for t in val_clips_labels_rot for c in t[0]]
    val_labels_rot_all = [c for t in val_clips_labels_rot for c in t[1]]
    val_clips_rot_all, val_labels_rot_all = equalize_posneg(val_clips_rot_all, val_labels_rot_all)
    val_clips_rot_all = val_clips_rot_all[0:-(len(val_clips_rot_all)%batch_size)]
    val_labels_rot_all = val_labels_rot_all[0:-(len(val_labels_rot_all)%batch_size)]
    pickle.dump(val_clips_rot_all, open(directory+"/pickles/val_clips_rot_all.pickle", 'wb'), protocol=2)
    pickle.dump(val_labels_rot_all, open(directory+"/pickles/val_labels_rot_all.pickle", 'wb'), protocol=2)
    return

def train_classify(num_epochs, net, num_hidden_nodes, directory=".", out_directory=".", restore=False, ckpt_directory=None):
    data_clips_labels = pickle.load(open(directory+"/pickles/data_clips_labels.pickle", 'rb'))
    train_clips_rot_all = pickle.load(open(directory+"/pickles/train_clips_rot_all.pickle", 'rb'))
    train_labels_rot_all = pickle.load(open(directory+"/pickles/train_labels_rot_all.pickle", 'rb'))
    val_clips_rot_all = pickle.load(open(directory+"/pickles/val_clips_rot_all.pickle", 'rb'))
    val_labels_rot_all = pickle.load(open(directory+"/pickles/val_labels_rot_all.pickle", 'rb'))
    batch_size = pickle.load(open(directory+"/pickles/batch_size.pickle", 'rb'))

    nn_best_val, nn_last_epoch = train_nn(train_clips_rot_all, train_labels_rot_all, val_clips_rot_all, val_labels_rot_all, 
                                          net, num_hidden_nodes, num_epochs, batch_size, prefix=out_directory, restore=restore, ckpt_directory=ckpt_directory)

    nn_labels_best_val = [nn_best_val(t[0]) for t in data_clips_labels]
    pickle.dump(nn_labels_best_val, open(out_directory+"/pickles/nn_labels_best_val.pickle",'wb'), protocol=2)
    if not restore:
        nn_labels_last_epoch = [nn_last_epoch(t[0]) for t in data_clips_labels]
        pickle.dump(nn_labels_last_epoch, open(out_directory+"/pickles/nn_labels_last_epoch.pickle", 'wb'), protocol=2)
    return


def labels_to_images(directory=".", clip_sz=15, clip_step=1, downscale_factor=2, restore=False):
    nn_labels_best_val = pickle.load(open(directory+"/pickles/nn_labels_best_val.pickle",'rb'))
    nn_pred_stk_best_val = [labels_to_stk(x, (512/downscale_factor, 512/downscale_factor), clip_sz, clip_step) for x in nn_labels_best_val] 
    pickle.dump(nn_pred_stk_best_val, open(directory+"/pickles/nn_pred_stk_best_val.pickle",'wb'), protocol=2)
    if not restore:
        nn_labels_last_epoch = pickle.load(open(directory+"/pickles/nn_labels_last_epoch.pickle", 'rb'))
        nn_pred_stk_last_epoch = [labels_to_stk(x, (512/downscale_factor, 512/downscale_factor), clip_sz, clip_step) for x in nn_labels_last_epoch] 
        pickle.dump(nn_pred_stk_last_epoch, open(directory+"/pickles/nn_pred_stk_last_epoch.pickle", 'wb'), protocol=2)
    return

def combine_stks(stks, weights):
    combined_stks = []
    for i in range(len(stks[0])):
        combined_stk = np.zeros(stks[0][i].shape)
        for j in range(len(stks)):
            combined_stk = combined_stk + weights[j] * stks[j][i]
        combined_stks.append(combined_stk)        
    return combined_stks

def stk_to_rois(stk, threshold, min_size, downscale_factor=2):
    thresholded_stk = (stk > threshold).astype(float)
    labels = measure.label(thresholded_stk, background=0)
    labels_set = set(labels.flatten())
    rois = []
    for label in labels_set:
        if label == 0: continue
        if np.sum((labels==label).astype(int)) < min_size: continue
        nroi = np.zeros((stk.shape[0], stk.shape[1]))
        cx,cy = np.where(labels==label)
        cx,cy = int(cx.mean()), int(cy.mean())
        x,y = np.ogrid[0:nroi.shape[0], 0:nroi.shape[1]]
        r = 4
        mask =  (cx-x)**2 + (cy-y)**2 <= r*r
        nroi[mask] = 1
        #nroi[labels==label] = 1
        rois.append(zoom(nroi, downscale_factor, order=0))
    rois = np.array(rois)
    #plt.figure()
    #plt.imshow(rois.max(axis=0), cmap='gray')
    #plt.show()
    return rois, thresholded_stk, labels

def stk_to_rois_new(stk, threshold, min_size, downscale_factor=2):
    thresholded_stk = (stk > threshold).astype(float)
    labels = measure.label(thresholded_stk, background=0)
    labels_set = set(labels.flatten())
    rois = []
    for label in labels_set:
        if label == 0: continue
        if np.sum((labels==label).astype(int)) < min_size: continue
        nroi = np.zeros((512, 512))
        cx,cy = np.where(labels==label)
        cx,cy = int(cx.mean()), int(cy.mean())
        x,y = np.ogrid[0:512, 0:512]
        r = 8
        mask =  ((cx*2)-x)**2 + ((cy*2)-y)**2 <= r*r
        nroi[mask] = 1
        #nroi[labels==label] = 1
        rois.append(nroi)
    rois = np.array(rois)
    #plt.figure()
    #plt.imshow(rois.max(axis=0), cmap='gray')
    #plt.show()
    return rois, thresholded_stk, labels

def find_postprocessing_hyperparameters(directory_list=["."], out_directory="."):
    threshold_grid = np.linspace(0.5,0.9,20) #[0.5, 0.9]
    min_size_grid = np.linspace(10, 15, 5) #[5, 30] 
    weights = [(1,0)] #[(i, 1-i) for i in np.linspace(0,1,20)]

    nn_pred_stks = [pickle.load(open(d+"/pickles/nn_pred_stk_best_val.pickle", 'rb')) for d in directory_list]
    val_nn_pred_stks = [t[8:10] for t in nn_pred_stks]
    val_raw_data = load_data()[8:10] #pickle.load(open(directory_list[0]+"/pickles/raw_data.pickle", 'rb'))[8:9]
    val_true_labels = [v[1] for v in val_raw_data]

    k = 0
    best_score = 0
    best_params = ([], 0, 0)
    for ws in weights: #itertools.product(*weights):
        #print ws
        combined_nn_pred_stks = combine_stks(val_nn_pred_stks, ws)
        for t, ms in itertools.product(threshold_grid, min_size_grid):
            print k
            print t, ms
            k += 1
            rois, _, _ = zip(*map(lambda s: stk_to_rois(s, t, ms), combined_nn_pred_stks))
            if rois[0].shape[0] == 0 or rois[1].shape[0] == 0: continue
            score = Score(None, None, val_true_labels, rois).total_f1_score
            print score
            if score > best_score:
                best_score = score
                #print "NEW BEST"
                best_params = (ws, t, ms)
    pickle.dump(best_params, open(out_directory+"/pickles/best_postprocessing_params.pickle", 'wb'), protocol=2)
    return
    
def images_to_score(directory_list=["."], out_directory=".", params_directory=".", params=None):
    if params is None:
        weights, threshold, min_size = pickle.load(open(params_directory+"/pickles/best_postprocessing_params.pickle", 'rb'))
    else:
        weights, threshold, min_size = params
    print weights, threshold, min_size
    nn_pred_stks = [pickle.load(open(d+"/pickles/nn_pred_stk_best_val.pickle", 'rb')) for d in directory_list]
    raw_data = load_data() #pickle.load(open(directory_list[0]+"/pickles/raw_data.pickle", 'rb'))
    true_labels = [r[1] for r in raw_data]
    combined_pred_stks = combine_stks(nn_pred_stks, weights)
    rois, thresholded_stks, stk_labels = zip(*map(lambda s: stk_to_rois(s, threshold, min_size), combined_pred_stks))
    pickle.dump(combined_pred_stks, open(out_directory+"/pickles/combined_pred_stks.pickle", 'wb'), protocol=2)
    pickle.dump(thresholded_stks, open(out_directory+"/pickles/thresholded_pred_stks.pickle", 'wb'), protocol=2)
    pickle.dump(stk_labels, open(out_directory+"/pickles/pred_stk_roi_labels.pickle", 'wb'), protocol=2)
    pickle.dump(rois, open(out_directory+"/pickles/final_rois.pickle", 'wb'), protocol=2)
    train_score = Score(None, None, true_labels[0:8], rois[0:8])
    val_score = Score(None, None, true_labels[8:10], rois[8:10])
    test_score = Score(None, None, true_labels[10:12], rois[10:12])
    print str(train_score)
    print
    print str(val_score)
    print
    print str(test_score)
    return test_score

"""
def main():
    if len(sys.argv) == 1:
        print "Usage: python {} clip_sz clip_step min_batch_size num_epochs prefix [load]".format(sys.argv[0])
        return
    clip_sz = int(sys.argv[1])
    clip_step = int(sys.argv[2])
    min_batch_size = int(sys.argv[3])
    num_epochs = int(sys.argv[4])
    prefix = '' if (len(sys.argv) <= 5) else sys.argv[5]
    load = True if (len(sys.argv) > 6 and sys.argv[6] == 'load') else False
    num_hidden_nodes = 10
    radius = 2
    net = onelayer_7x7

    data_raw = load_data()
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
        nn_best_val, nn_last_epoch = train_nn(train_clips_rot_all, train_labels_rot_all, val_clips_rot_all, val_labels_rot_all, net, num_hidden_nodes, num_epochs, batch_size, prefix=prefix)
        # classify data
        train_nn_labels_best_val = [nn_best_val(t[0]) for t in train_clips_labels] 
        val_nn_labels_best_val = [nn_best_val(t[0]) for t in val_clips_labels] 
        test_nn_labels_best_val = [nn_best_val(t[0]) for t in test_clips_labels]
        train_nn_labels_last_epoch = [nn_last_epoch(t[0]) for t in train_clips_labels] 
        val_nn_labels_last_epoch = [nn_last_epoch(t[0]) for t in val_clips_labels] 
        test_nn_labels_last_epoch = [nn_last_epoch(t[0]) for t in test_clips_labels]
        # save results
        pickle.dump(train_nn_labels_best_val, open(prefix+"nn_train_labels_best_val.pickle",'wb'))
        pickle.dump(val_nn_labels_best_val, open(prefix+"nn_val_labels_best_val.pickle", 'wb'))
        pickle.dump(test_nn_labels_best_val, open(prefix+"nn_test_labels_best_val.pickle", 'wb'))
        pickle.dump(train_nn_labels_last_epoch, open(prefix+"nn_train_labels_last_epoch.pickle",'wb'))
        pickle.dump(val_nn_labels_last_epoch, open(prefix+"nn_val_labels_last_epoch.pickle", 'wb'))
        pickle.dump(test_nn_labels_last_epoch, open(prefix+"nn_test_labels_last_epoch.pickle", 'wb'))
    
    else:
        train_nn_labels_best_val = pickle.load(open(prefix+"nn_train_labels_best_val.pickle",'rb'))
        val_nn_labels_best_val = pickle.load(open(prefix+"nn_val_labels_best_val.pickle", 'rb'))
        test_nn_labels_best_val = pickle.load(open(prefix+"nn_test_labels_best_val.pickle", 'rb'))
        train_nn_labels_last_epoch = pickle.load(open(prefix+"nn_train_labels_last_epoch.pickle",'rb'))
        val_nn_labels_last_epoch = pickle.load(open(prefix+"nn_val_labels_last_epoch.pickle", 'rb'))
        test_nn_labels_last_epoch = pickle.load(open(prefix+"nn_test_labels_last_epoch.pickle", 'rb'))

    # convert predictions back to 1/0 arrays
    actual_train_labels = [t[1] for t in data_raw[0:8]]
    actual_test_labels = [t[1] for t in data_raw[10:12]]
    actual_val_labels = [t[1] for t in data_raw[8:10]]
    #threshold, eps, min_samples, final_radius = train_threshold_hyperparameters(val_nn_labels_best_val[0], actual_val_labels[0], clip_sz, clip_step)
    threshold, eps, min_samples, final_radius = (0.9, 0.0147, 10, 7.8)#(0.0254, 72, 7.8)
    #threshold = None
    print "threshold: {}\neps: {}\nmin_samples: {}\nradius: {}\n".format(threshold, eps, min_samples, final_radius)
    train_nn_pred_stk_best_val = [labels_to_stk(x, (512/DOWNSCALE_FACTOR, 512/DOWNSCALE_FACTOR), clip_sz, clip_step, threshold) for x in train_nn_labels_best_val] 
    val_nn_pred_stk_best_val = [labels_to_stk(x, (512/DOWNSCALE_FACTOR, 512/DOWNSCALE_FACTOR), clip_sz, clip_step, threshold) for x in val_nn_labels_best_val]
    test_nn_pred_stk_best_val = [labels_to_stk(x, (512/DOWNSCALE_FACTOR, 512/DOWNSCALE_FACTOR), clip_sz, clip_step, threshold) for x in test_nn_labels_best_val]
    train_nn_pred_stk_last_epoch = [labels_to_stk(x, (512/DOWNSCALE_FACTOR, 512/DOWNSCALE_FACTOR), clip_sz, clip_step, threshold) for x in train_nn_labels_last_epoch] 
    val_nn_pred_stk_last_epoch = [labels_to_stk(x, (512/DOWNSCALE_FACTOR, 512/DOWNSCALE_FACTOR), clip_sz, clip_step, threshold) for x in val_nn_labels_last_epoch]
    test_nn_pred_stk_last_epoch = [labels_to_stk(x, (512/DOWNSCALE_FACTOR, 512/DOWNSCALE_FACTOR), clip_sz, clip_step, threshold) for x in test_nn_labels_last_epoch]

    
    # plot things
    thresh = "_thresh" if threshold is not None else ""
    plt.figure()
    plt.imshow(train_data[0][0].squeeze(), cmap="gray")
    plt.savefig(prefix+"nn_train_raw.png")
    plt.figure()
    plt.imshow(train_data[0][1], cmap="gray")
    plt.savefig(prefix+"nn_train_actual.png")
    plt.figure()
    plt.imshow(train_nn_pred_stk_best_val[0], cmap="gray")
    plt.savefig(prefix+"nn_train_pred_best_val"+str(thresh)+".png")
    plt.figure()
    plt.imshow(train_nn_pred_stk_last_epoch[0], cmap="gray")
    plt.savefig(prefix+"nn_train_pred_last_epoch"+str(thresh)+".png")
   
    plt.figure()
    plt.imshow(test_data[0][0].squeeze(), cmap="gray")
    plt.savefig(prefix+"nn_test0_raw.png")
    plt.figure()
    plt.imshow(test_data[0][1], cmap="gray")
    plt.savefig(prefix+"nn_test0_actual.png")
    plt.figure()
    plt.imshow(test_nn_pred_stk_best_val[0], cmap="gray")
    plt.savefig(prefix+"nn_test0_pred_best_val"+str(thresh)+".png")
    plt.figure()
    plt.imshow(test_nn_pred_stk_last_epoch[0], cmap="gray")
    plt.savefig(prefix+"nn_test0_pred_last_epoch"+str(thresh)+".png")

    plt.figure()
    plt.imshow(test_data[1][0].squeeze(), cmap="gray")
    plt.savefig(prefix+"test1_raw.png")
    plt.figure()
    plt.imshow(test_data[1][1], cmap="gray")
    plt.savefig(prefix+"nn_test1_actual.png")
    plt.figure()
    plt.imshow(test_nn_pred_stk_best_val[1], cmap="gray")
    plt.savefig(prefix+"nn_test1_pred_best_val"+str(thresh)+".png")
    plt.figure()
    plt.imshow(test_nn_pred_stk_last_epoch[1], cmap="gray")
    plt.savefig(prefix+"nn_test1_pred_last_epoch"+str(thresh)+".png")
    
    # convert stacked predictions to final ROI format
    train_nn_pred_final = [nn_stk_pred_to_final_roi_format(x, eps, min_samples, final_radius) for x in train_nn_pred_stk_best_val]
    val_nn_pred_final = [nn_stk_pred_to_final_roi_format(x, eps, min_samples, final_radius) for x in val_nn_pred_stk_best_val]
    test_nn_pred_final = [nn_stk_pred_to_final_roi_format(x, eps, min_samples, final_radius) for x in test_nn_pred_stk_best_val]

    # get final score
    #print actual_train_labels[0].shape
    #print train_nn_pred_final[0].shape
    #print actual_test_labels[0].shape
    #print test_nn_pred_final[0].shape
    train_score = Score(None, None, actual_train_labels, train_nn_pred_final)
    test_score = Score(None, None, actual_test_labels, test_nn_pred_final)
    test0_score = Score(None, None, actual_test_labels[0:1], test_nn_pred_final[0:1])
    test1_score = Score(None, None, actual_test_labels[1:2], test_nn_pred_final[1:2])
    print str(train_score)
    print
    print str(test_score)
    print
    print str(test0_score)
    print
    print str(test1_score)

    # plot things
    train_score.plot()
    test_score.plot()
    test0_score.plot()
    test1_score.plot()

    plt.show()
    return

    plt.figure()
    plt.imshow(train_nn_pred_final[0].max(axis=0), cmap="gray")
    plt.savefig(prefix+"nn_train_final.png")

    plt.figure()
    plt.imshow(test_nn_pred_final[0].max(axis=0), cmap="gray")
    plt.savefig(prefix+"nn_test_final.png")
    
    # show all the plots!
    plt.show()
    

if __name__ == "__main__":
    main()
"""
