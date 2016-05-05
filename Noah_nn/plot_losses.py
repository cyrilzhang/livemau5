import matplotlib.pyplot as plt
import pickle
import sys
import numpy as np

def plot_losses(d, parse=False, parse_file=""):
    if parse:
        train_loss_per_epoch, val_loss_per_epoch = parse_loss(parse_file)
    else:
        train_loss_per_epoch = pickle.load(open(d+"/pickles/train_loss.pickle", 'rb'))
        val_loss_per_epoch = pickle.load(open(d+"/pickles/val_loss.pickle", 'rb'))
    print "best val loss: {}".format(np.min(val_loss_per_epoch))
    plot(d, train_loss_per_epoch, val_loss_per_epoch)

def plot(d, train_loss_per_epoch, val_loss_per_epoch):
    fig, ax = plt.subplots(2, sharex=True)
    ax[0].plot(range(len(train_loss_per_epoch)), train_loss_per_epoch, label="Training Loss",c='b', lw=2)
    ax[1].plot(range(len(val_loss_per_epoch)), val_loss_per_epoch, label="Validation Loss", c='g', lw=2)
    ax[0].set_xlim((0,len(train_loss_per_epoch)-1))
    ax[1].set_xlim((0,len(val_loss_per_epoch)-1))
    ax[0].set_ylim((0,0.7))
    ax[1].set_ylim((0,0.7))
    ax[0].legend()
    ax[1].legend(loc = 'lower right')
    #ax[0].set_xlabel("Epoch")
    ax[1].set_xlabel("Epoch")
    ax[0].set_ylabel("Mean Cross-Entropy Loss")
    ax[1].set_ylabel("Mean Cross-Entropy Loss")
    fig.subplots_adjust(hspace=.5)
    plt.savefig(d+"loss_curves_nn.png")


def parse_loss(d):
    f = open(d, 'r')
    train_loss_per_epoch = []
    val_loss_per_epoch = []
    for line in f:
        if not line.startswith("Training Loss"): continue
        p = line.partition("Validation Loss: ")
        val_loss = float(p[2][0:6])
        t = line.partition("Training Loss: ")
        train_loss = float(t[2][0:6])
        train_loss_per_epoch.append(train_loss)
        val_loss_per_epoch.append(val_loss)
    return train_loss_per_epoch, val_loss_per_epoch


if __name__ == "__main__":
    d = sys.argv[1]
    parse = True if len(sys.argv) > 1  else False
    parse_file = sys.argv[2] if parse else ""
    plot_losses(d, parse, parse_file)
