import argparse
import os
import torch
from data.dataloader import get_data_loader
from utils.utils import load_checkpoint, read_model_params, plot_metrics
from utils.utils import evaluate_matches, evaluate_matches, vis_fg_bg
from models.fb_bg_classifier import FgBgClassifier
import numpy as np
import torch.nn as nn

import warnings
warnings.filterwarnings("ignore")


def infer(opt):
    #get dataloader
    dataloader = get_data_loader(opt.annotations_dir, opt.seg_dir,
                                 False, 1, False)
    #

    #load model
    model_params = read_model_params(opt.params_path)
    model = FgBgClassifier(model_params).to(opt.device)
    load_checkpoint(opt.checkpoint_epoch, opt.checkpoint_dir, model)
    model.eval()
    #

    mts = np.arange(1, 11)*0.1
    #TODO should these be averaged or total
    tps = [None]*len(mts)
    fps = [None]*len(mts)
    fns = [None]*len(mts)

    with torch.no_grad():
        for _, data in enumerate(dataloader):
            if not len(data) == 1:
                raise RuntimeError('Only batch size 1 supported in test')
            
            box_features, keypoint_vecs, is_tags, scores, is_fgs, gt_vis = data[0][0]
            preds = model(box_features, keypoint_vecs, is_tags, scores)
            preds = nn.functional.sigmoid(preds)

            for mt_ind in range(len(mts)):
                mt = mts[mt_ind]
                tp, fp, fn = evaluate_matches(preds, is_fgs, mt)

                if tps[mt_ind] == None:
                    tps[mt_ind] = []
                    fps[mt_ind] = []
                    fns[mt_ind] = []
                     
                tps[mt_ind].append(tp)
                fps[mt_ind].append(fp)
                fns[mt_ind].append(fn)

            _, gt_centers, image_path, basename = gt_vis
            output_path = os.path.join(opt.vis_dir, basename.replace('.json', '.png'))
            vis_fg_bg(preds, is_fgs, image_path, gt_centers, opt.vis_fg_thresh, opt.vis_radius, output_path)
            
    precisions = [None]*len(mts)
    recalls = [None]*len(mts)
    for mt_ind in range(len(mts)):
        tps_sum = np.sum(tps[mt_ind])
        if tps_sum == 0:
            precisions[mt_ind] = 0
            recalls[mt_ind] = 0
        else:
            precisions[mt_ind] = tps_sum / (tps_sum + np.sum(fps[mt_ind]))
            recalls[mt_ind] = tps_sum / (tps_sum + np.sum(fns[mt_ind]))

        precisions[mt_ind] = np.mean(precisions[mt_ind]) 
        recalls[mt_ind] = np.mean(recalls[mt_ind]) 

    plot_metrics(opt.vis_dir, precisions, recalls, mts)

default_image_dir = '/home/frc-ag-3/harry_ws/fruitlet_2023/labelling/inhand/fg_bg_images'
def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotations_dir', required=True)
    parser.add_argument('--seg_dir', required=True)

    parser.add_argument('--params_path', default='./params/default_params.yml')

    parser.add_argument('--checkpoint_dir', default='./checkpoints')
    parser.add_argument('--checkpoint_epoch', type=int, required=True)

    parser.add_argument('--vis_dir', default='./vis')
    parser.add_argument('--image_dir', default=default_image_dir)
    parser.add_argument('--vis_radius', type=int, default=2)
    parser.add_argument('--vis_fg_thresh', type=float, default=0.8)

    parser.add_argument('--device', default='cuda')
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    opt = parse_args()

    infer(opt)