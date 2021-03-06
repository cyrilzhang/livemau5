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
from scipy.misc import imresize

IMG_CHANNELS = 1
DOWNSCALE_FACTOR = 2

def load_data():
    # generate filenames
    files = []
    for i in range(1,4):
        for j in range(1,5):
            prefix = '../data/AMG%d_exp%d'%(i,j)
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
        low_p = np.percentile(stk.flatten(), 3)
        high_p = np.percentile(stk.flatten(), 99)
        #print np.min(stk.flatten()), low_p, high_p, np.max(stk.flatten())
        
#        plt.figure()
#        plt.hist(stk.flatten())
#        plt.figure()
#        plt.imshow(downscale_local_mean(stk[0], (DOWNSCALE_FACTOR, DOWNSCALE_FACTOR)), cmap="gray")
#        plt.figure()
#        plt.imshow(downscale_local_mean(np.mean(stk, axis=0), (DOWNSCALE_FACTOR, DOWNSCALE_FACTOR)), cmap="gray")

        stk = np.clip(stk, low_p, high_p)
        stk = stk - stk.mean()
        stk = np.divide(stk-np.min(stk), np.max(stk) - np.min(stk))
        #print np.min(stk.flatten()),  np.max(stk.flatten())

#        plt.figure()
#        plt.hist(stk.flatten())

        # Downscale
        stk = downscale_local_mean(stk, (1,DOWNSCALE_FACTOR,DOWNSCALE_FACTOR))
        roi = downscale_local_mean(roi, (1,DOWNSCALE_FACTOR,DOWNSCALE_FACTOR))
        new_stk = np.zeros((stk.shape[1], stk.shape[2], IMG_CHANNELS))
        # Uncomment following 2 lines to use more channels
        #new_stk[:,:,2] = stk.max(axis=0)
        #new_stk[:,:,1] = np.std(stk, axis=0)
        new_stk[:,:,0] = np.mean(stk, axis=0)

#        plt.figure()
#        plt.imshow(stk[0], cmap="gray")
#        plt.figure()
#        plt.imshow(new_stk[:,:,0], cmap="gray")
#        plt.show()

        roi_centroids = get_centroids(roi, radius).max(axis=0)
        data[i] = (new_stk, roi.max(axis=0), roi.max(axis=0))#(new_stk, roi_centroids, roi.max(axis=0))
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

def labels_to_stk(labels, orig_shape, sz, step, threshold=None):
    rows, cols = orig_shape
    stk = np.zeros((rows, cols)) #if threshold is not None else np.zeros((rows, cols))
    i = 0
    max_rows = int(rows-sz/2) if sz%2 == 0 else int(rows-sz/2)-1
    max_cols = int(cols-sz/2) if sz%2 == 0 else int(cols-sz/2)-1
    for r,c in itertools.product(range(int(sz/2), max_rows, step), 
                                 range(int(sz/2), max_cols, step)):
        assert(len(labels[i].shape) == 1 and (labels[i].shape)[0] == 2)
        if threshold is not None:
            stk[r,c] = 1 if np.argmax(labels[i]) == 1 and labels[i][1] > threshold else 0 
        else:
            stk[r,c] = labels[i][1] - labels[i][0]#np.array([labels[i][1], 0, labels[i][0]])
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
    
def train_threshold_hyperparameters(pred, actual, clip_sz, clip_step):
    best_parameters = (0,0,0,0)
    best_score = 0
    threshold_test = np.linspace(0.4, 0.9, 5)
    eps_test = np.logspace(-3.5,-0.5,10) #[0.00794] #np.logspace(-2.5,-1.5,6)
    min_samples_test = np.linspace(10,50,5)
    radius_test = [7.6] #np.linspace(3,15,6)
    i = 0
    for threshold, eps, min_samples, radius in itertools.product(threshold_test, eps_test, min_samples_test, radius_test):
        stk = labels_to_stk(pred, (512/DOWNSCALE_FACTOR, 512/DOWNSCALE_FACTOR), clip_sz, clip_step, threshold) 
        predictions = nn_stk_pred_to_final_roi_format(stk, eps, min_samples, radius)
        s = Score(None, None, [actual], [predictions])
        score = s.total_f1_score
        print "{}, {}, {}: Score: {}".format(eps, min_samples, radius, score)
        if score > best_score:
            best_parameters = (threshold, eps, min_samples, radius)
            best_score = score
    print "Best Score: {}".format(best_score)
    return best_parameters

def nn_stk_pred_to_points(labels):
    pts = []
    for x in range(labels.shape[0]):
        for y in range(labels.shape[1]):
            if labels[x,y] == 1:
                pts.append((x/(512./DOWNSCALE_FACTOR), y/(512./DOWNSCALE_FACTOR)))
    return pts

def nn_stk_pred_to_final_roi_format(pred, eps, min_samples, radius):
    pred = pred.squeeze()
    pred_pts = nn_stk_pred_to_points(pred)
    dbscan = DBSCAN(metric='euclidean', eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(pred_pts)
    print set(clusters)

    
    #plt.figure()
    #plt.imshow(pred, cmap="gray")
    #plt.figure()
    #x,y = zip(*pred_pts)
    #plt.scatter(map(lambda x: x*256, y),map(lambda x: (1-x)*256, x), marker='.', lw=0, c=clusters)
    #plt.xlim((0,256))
    #plt.ylim((0,256))
    

    z = zip(clusters, pred_pts)
    centroids = []
    for s in set(clusters):
        if s == -1: continue
        pts_in_cluster = filter(lambda x: x[0] == s, z)
        cluster, pts = zip(*pts_in_cluster)
        xs, ys = zip(*pts)
        xmean, ymean = np.mean(xs), np.mean(ys)
        centroids.append((xmean*(512),ymean*(512)))
    final_predictions = np.zeros((len(centroids),(512),(512)))
    for i,(cx,cy) in enumerate(centroids):
        for x in range(512):
            for y in range(512):
                if abs(x-cx) > radius or abs(y-cy) > radius: continue
                elif np.sqrt((y-cy)**2 + (x-cx)**2) <= radius:
                    final_predictions[i,x,y] = 1

    #plt.figure()
    #plt.imshow(final_predictions.max(axis=0), cmap="gray")
    #plt.xlim((0,512))
    #plt.ylim((0,512))
    #plt.show()

    return final_predictions


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
