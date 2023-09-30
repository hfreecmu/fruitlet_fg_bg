import os
import json
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt

#read json
def read_dict(path):
    with open(path) as f:
        data = json.load(f)
    return data

#read pkl file
def read_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)
    
#save model checkpoint
def save_checkpoint(epoch, checkpoint_dir, model):
    model_path = os.path.join(checkpoint_dir, 'epoch_%d.pth' % epoch)
    torch.save(model.state_dict(), model_path)

#load model checkpoint
def load_checkpoint(epoch, checkpoint_dir, model):
    model_path = os.path.join(checkpoint_dir, 'epoch_%d.pth' % epoch)
    model.load_state_dict(torch.load(model_path))

#plot and save scores
def plot_and_save_scores(checkpoint_dir, epochs, precisions, recalls, f1_scores):
    epochs = np.array(epochs)
    precisions = np.array(precisions)
    recalls = np.array(recalls)
    f1_scores = np.array(f1_scores)

    scores = np.vstack((epochs, precisions, recalls, f1_scores))
    scores_np_path = os.path.join(checkpoint_dir, 'scores.npy')
    np.save(scores_np_path, scores)

    scoress_plt_plath = os.path.join(checkpoint_dir, 'scores.png')

    plt.plot(epochs, precisions, 'b')
    plt.plot(epochs, recalls, 'r')
    plt.plot(epochs, f1_scores, 'g')
    plt.savefig(scoress_plt_plath)

    plt.clf()

#plot and save loss
def plot_and_save_loss(checkpoint_dir, epochs, losses, is_train):
    epochs = np.array(epochs)
    losses = np.array(losses)

    if is_train:
        losses_np_path = os.path.join(checkpoint_dir, 'train_losses.npy')
        losses_plt_plath = os.path.join(checkpoint_dir, 'train_losses.png')
    else:
        losses_np_path = os.path.join(checkpoint_dir, 'losses.npy')
        losses_plt_plath = os.path.join(checkpoint_dir, 'losses.png')

    np.save(losses_np_path, np.vstack((epochs, losses)))

    plt.plot(epochs, losses, 'r')
    plt.savefig(losses_plt_plath)

    plt.clf()