import os
import json
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
import yaml
import cv2

from detectron2.config import get_cfg
from detectron2 import model_zoo

#read json
def read_dict(path):
    with open(path) as f:
        data = json.load(f)
    return data

#read yaml file
def read_yaml(path):
    with open(path, 'r') as f:
        yaml_to_read = yaml.safe_load(f)
                                
    return yaml_to_read

#read pkl file
def read_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)
    
#write_pickle
def write_pickle(path, data):
    with open(path, "wb") as f:
        pickle.dump(data, f)
    
#save model checkpoint
def save_checkpoint(epoch, checkpoint_dir, model):
    model_path = os.path.join(checkpoint_dir, 'epoch_%d.pth' % epoch)
    torch.save(model.state_dict(), model_path)

#load model checkpoint
def load_checkpoint(epoch, checkpoint_dir, model):
    model_path = os.path.join(checkpoint_dir, 'epoch_%d.pth' % epoch)
    model.load_state_dict(torch.load(model_path))

#read model params
def read_model_params(params_path):
    params = read_yaml(params_path)
    return params

#plot and save scores
def plot_and_save_scores(checkpoint_dir, epochs, precisions, recalls):
    epochs = np.array(epochs)
    precisions = np.array(precisions)
    recalls = np.array(recalls)

    scores = np.vstack((epochs, precisions, recalls))
    scores_np_path = os.path.join(checkpoint_dir, 'scores.npy')
    np.save(scores_np_path, scores)

    scoress_plt_plath = os.path.join(checkpoint_dir, 'scores.png')

    plt.plot(epochs, precisions, 'b')
    plt.plot(epochs, recalls, 'r')
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

#create feature pred cfg
def create_cfg(model_file, score_thresh):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_file 
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[16, 32, 64, 128]]
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.0, 2.0]]
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

    cfg.INPUT.MIN_SIZE_TEST = 1080
    cfg.INPUT.MAX_SIZE_TEST = 1440

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh

    return cfg

#evaluate matches
def evaluate_matches(preds, is_fgs, fg_thresh):
    preds = preds.detach().cpu().numpy().reshape((-1))
    is_fgs = is_fgs.detach().cpu().numpy().reshape((-1))

    if not preds.shape == is_fgs.shape:
        raise RuntimeError('Why shapes do not match')

    tp = 0
    fp = 0
    fn = 0

    for i in range(preds.shape[0]):
        pred = preds[i]
        is_fg = is_fgs[i]

        if is_fg > fg_thresh:
            if pred > fg_thresh:
                tp += 1
            else:
                fn += 1
        elif pred > fg_thresh:
            fp += 1

    return tp, fp, fn

def vis_fg_bg(preds, is_fgs, image_path, gt_centers, fg_thresh, vis_radius, output_path):
    
    im = cv2.imread(image_path)
    preds = preds.detach().cpu().numpy().reshape((-1))
    is_fgs = is_fgs.detach().cpu().numpy().reshape((-1))

    if not preds.shape == is_fgs.shape:
        raise RuntimeError('Why shapes do not match')
    
    for i in range(preds.shape[0]):
        pred = preds[i]
        is_fg = is_fgs[i]

        if pred < fg_thresh:
            continue

        if is_fg < fg_thresh:
            color = [0, 0, 255]
        else:
            color = [0, 255, 0]

        y0, x0 = gt_centers[i]
        cv2.circle(im, (int(x0), int(y0)), vis_radius, color, -1)

    cv2.imwrite(output_path, im)

def plot_metrics(output_dir, precisions, recalls, match_thresholds):
    comb_path = os.path.join(output_dir, 'metrics.png')
    comp_np_path = comb_path.replace('.png', '.npy')

    f1_scores = [None]*len(precisions)
    for i in range(len(precisions)):
        if precisions[i] == 0 or recalls[i] == 0:
            f1_scores[i] = 0
        else:
            f1_scores[i] = 2*precisions[i]*recalls[i] / (precisions[i] + recalls[i])

    plt.plot(match_thresholds, precisions, 'b', label="precision")
    plt.plot(match_thresholds, recalls, 'r', label="recall")
    plt.plot(match_thresholds, f1_scores, 'g', label="f1")
    plt.legend(loc="lower left")
    plt.xlabel("Matching Thresholds")
    plt.xticks(np.arange(min(match_thresholds), max(match_thresholds), 0.1))
    plt.savefig(comb_path)

    plt.clf()

    comb_np = np.zeros((len(match_thresholds), 3))
    comb_np[:, 0] = match_thresholds
    comb_np[:, 1] = precisions
    comb_np[:, 2] = recalls
    np.save(comp_np_path, comb_np)
