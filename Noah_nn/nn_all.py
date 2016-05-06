import numpy as np
import matplotlib.pyplot as plt
import sys
import cPickle as pickle
from skimage.transform import downscale_local_mean
import itertools
import os.path
from matplotlib.patches import Rectangle
from random import sample, shuffle
from skimage import measure
from scipy.ndimage import zoom
from scipy import ndimage as ndi
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from skimage.morphology import remove_small_objects
from load import *
from test import *
from nn_arch import *
from nn import train_nn

########################
# Helper Functions
########################

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

def combine_stks(stks, weights):
    combined_stks = []
    for i in range(len(stks[0])):
        combined_stk = np.zeros(stks[0][i].shape)
        for j in range(len(stks)):
            combined_stk = combined_stk + weights[j] * stks[j][i]
        combined_stks.append(combined_stk)        
    return combined_stks

def stk_to_rois(stk, threshold, min_size, max_window=8, downscale_factor=2):
    thresholded_stk = stk > threshold
    thresholded_stk = remove_small_objects(thresholded_stk, min_size)
    distance = ndi.distance_transform_edt(thresholded_stk)
    cropped_stk = stk.copy()
    cropped_stk[np.logical_not(thresholded_stk)] = 0
    combined_stk = cropped_stk + distance/distance.max()
    local_max = peak_local_max(combined_stk, indices=False, 
                               footprint=np.ones((max_window, max_window)), 
                               labels=thresholded_stk)
    markers = ndi.label(local_max)[0]
    labels = watershed(-combined_stk, markers, mask=thresholded_stk)
    new_markers = markers.copy()
    for i in set(labels.flatten()):
        if i == 0: continue
        if np.sum(labels==i) < min_size:
            new_markers[markers==i] = 0
    labels = watershed(-combined_stk, new_markers, mask=thresholded_stk)
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
    return rois, thresholded_stk, labels

def stk_to_rois_original(stk, threshold, min_size, downscale_factor=2):
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

def stk_to_rois_test(stk, threshold, min_size, downscale_factor=2):
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

###################################
# Pipeline Functions 
###################################

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
