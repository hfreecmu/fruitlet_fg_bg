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

#pass through network
def predict(classifier, cloud_path, label_path, opt, shuffle, augment):
    np_clouds, tag_np_cloud = read_pickle(cloud_path)
    gt_labels = read_dict(label_path)
    
    torch_clouds = []
    is_tag = []
    target = []

    for cloud_ind in range(len(np_clouds)):
        np_cloud = np_clouds[cloud_ind]

        if np_cloud.shape[0] < opt.min_area:
            continue

        cloud_inds = np.random.choice(np_cloud.shape[0], size=opt.min_area, replace=False)
        np_cloud = np_cloud[cloud_inds]

        gt_val = gt_labels[cloud_ind][-1] * 1.0

        torch_cloud = torch.from_numpy(np_cloud).float()
        torch_cloud = torch_cloud.unsqueeze(0)
        torch_cloud = torch_cloud.permute(0, 2, 1)
        torch_cloud = torch_cloud.to(opt.device)
        
        torch_clouds.append(torch_cloud)
        is_tag.append(0.0)
        target.append(gt_val)

    torch_tag_cloud = torch.from_numpy(tag_np_cloud).float()
    torch_tag_cloud = torch_tag_cloud.unsqueeze(0)
    torch_tag_cloud = torch_tag_cloud.permute(0, 2, 1)
    torch_tag_cloud = torch_tag_cloud.to(opt.device)
    torch_clouds.append(torch_tag_cloud)
    is_tag.append(1.0)
    target.append(False)

    ###shuffle
    if shuffle:
        num_nodes = len(torch_clouds)
        rand_inds = np.random.choice(num_nodes, size=num_nodes, replace=False)
        rand_torch_clouds = []
        rand_is_tag = []
        rand_target = []
        for rand_ind in rand_inds:
            rand_torch_clouds.append(torch_clouds[rand_ind])
            rand_is_tag.append(is_tag[rand_ind])
            rand_target.append(target[rand_ind])

        torch_clouds = rand_torch_clouds
        is_tag = rand_is_tag
        target = rand_target
    ###

    is_tag = torch.as_tensor(is_tag, dtype=torch.float, device=opt.device)
    target = torch.as_tensor(target, dtype=torch.float, device=opt.device)
    pred = classifier(torch_clouds, is_tag).flatten()

    return pred, is_tag, target