import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from data.dataloader import get_data_loader
from models.fb_bg_classifier import FgBgClassifier
from utils.utils import save_checkpoint, read_model_params
from utils.utils import plot_and_save_scores, plot_and_save_loss, evaluate_matches

import warnings
warnings.filterwarnings("ignore")

#no cudnn memory issue
torch.backends.cudnn.enabled = False

def train(opt):
    #get dataloaders
    train_dataloader = get_data_loader(opt.annotations_dir, opt.seg_dir,
                                       True, opt.batch_size, opt.shuffle)
    val_dataloader = get_data_loader(opt.val_dir, opt.seg_dir,
                                       False, 1, False)
    #

    #load model
    model_params = read_model_params(opt.params_path)
    model = FgBgClassifier(model_params).to(opt.device)
    model.train()
    #

    #get optimizer
    optimizer = optim.Adam(model.parameters(), opt.lr, weight_decay=opt.weight_decay)
    #optimizer = optim.Adam(model.parameters(), opt.lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.milestones, gamma=opt.gamma)
    
    #loss
    pos_weight = opt.pos_weight*torch.ones((1)).to(opt.device)
    bce_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    #
    
    loss_array = []
    plot_epochs = []

    #for plotting
    loss_array = []
    plot_epochs = []

    precision_array = []
    recall_array = []
    val_loss_array = []
    #

    #TODO really this should be done in a dataloader
    #so that we can augment and what not
    for epoch in range(opt.num_epochs):
        losses = []
        for _, data in enumerate(train_dataloader):
            loss = []
            optimizer.zero_grad()
            for data_num in range(len(data)):
                box_features, keypoint_vecs, is_tags, scores, is_fgs, _ = data[data_num][0]
            
                preds = model(box_features, keypoint_vecs, is_tags, scores)
                fg_bg_loss = bce_loss_fn(preds, is_fgs.reshape((-1, 1)))

                loss.append(fg_bg_loss)

            loss = torch.mean(torch.stack(loss))
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        scheduler.step()

        epoch_loss = np.mean(losses)
        print('loss for epoch', epoch, 'is: ', epoch_loss)

        log_epoch = epoch + 1
        if (log_epoch% opt.log_steps) == 0:
            #start by saving model and training losses
            save_checkpoint(epoch + 1, opt.checkpoint_dir, model)
            loss_array.append(epoch_loss)
            plot_epochs.append(log_epoch)
            plot_and_save_loss(opt.checkpoint_dir, plot_epochs, loss_array, True)

            #now evaluate
            model.eval()
            with torch.no_grad():
                tps = []
                fps = []
                fns = []
                val_losses = []

                for _, data in enumerate(val_dataloader):
                    if not len(data) == 1:
                        raise RuntimeError('Only batch size 1 supported validate')
                    
                    box_features, keypoint_vecs, is_tags, scores, is_fgs, _ = data[0][0]
                    preds = model(box_features, keypoint_vecs, is_tags, scores)
                    val_loss = bce_loss_fn(preds, is_fgs.reshape((-1, 1)))

                    preds = nn.functional.sigmoid(preds)
                    tp, fp, fn = evaluate_matches(preds, is_fgs, opt.fg_thresh)

                    tps.append(tp)
                    fps.append(fp)
                    fns.append(fn)
                    val_losses.append(val_loss.item())
            
            model.train()

            val_epoch_loss = np.mean(val_losses)
            print('val loss for epoch', epoch, 'is: ', val_epoch_loss)

            tps = np.sum(tps)
            fps = np.sum(fps)
            fns = np.sum(fns)

            if tps == 0:
                precision = 0
                recall = 0
            else:
                precision = tps / (tps + fps)
                recall = tps / (tps + fns)

            precision_array.append(precision)
            recall_array.append(recall)
            val_loss_array.append(val_epoch_loss)

            plot_and_save_scores(opt.checkpoint_dir, plot_epochs, precision_array,
                                 recall_array)
            plot_and_save_loss(opt.checkpoint_dir, plot_epochs, val_loss_array, False)
            


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotations_dir', required=True)
    parser.add_argument('--seg_dir', required=True)
    parser.add_argument('--val_dir', required=True)
    parser.add_argument('--params_path', default='./params/default_params.yml')

    parser.add_argument('--checkpoint_dir', default='./checkpoints')

    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--milestones', type=list, default=[20, 40, 100])
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--pos_weight', type=float, default=3.0)

    parser.add_argument('--log_steps', type=int, default=10)
    parser.add_argument('--fg_thresh', type=float, default=0.5)

    parser.add_argument('--shuffle', action='store_false')
    parser.add_argument('--device', default='cuda')
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    opt = parse_args()

    train(opt)