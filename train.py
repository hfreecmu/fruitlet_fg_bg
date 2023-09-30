import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.spatial.transform import Rotation
from data.dataloader import get_data_loader
from models.fb_bg_classifier import FgBgClassifier
from utils.utils import read_pickle, read_dict, save_checkpoint
from utils.utils import plot_and_save_scores, plot_and_save_loss

def train(opt):
    dataloader = get_data_loader(opt.cloud_dir, opt.label_dir, 
                                 opt.batch_size, opt.shuffle)
    loss_array = []
    plot_epochs = []

    #for eval
    val_dataloader = get_data_loader(opt.val_dir, opt.label_dir, 
                                     1, False)
    precision_array = []
    recall_array = []
    f1_array = []
    val_loss_array = []
    #

    classifier = FgBgClassifier().to(opt.device)
    classifier.train()

    optimizer = optim.Adam(classifier.parameters(), opt.lr, weight_decay=1e-4)

    milestones = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)

    pos_weight = opt.pos_weight*torch.ones((1)).to(opt.device)
    bce_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    #TODO really this should be done in a dataloader
    #so that we can augment and what not
    for epoch in range(opt.num_epochs):
        losses = []
        for _, data in enumerate(dataloader):
            loss = []
            optimizer.zero_grad()
            cloud_paths, label_paths = data
            for data_num in range(len(cloud_paths)):
                x_shift = np.random.uniform(-0.1, 0.1)
                y_shift = np.random.uniform(-0.1, 0.1)
                z_shift = np.random.uniform(-0.1, 0.1)
                rand_t = np.array([x_shift, y_shift, z_shift])
                rand_R = np.eye(3)
                # rand_R = Rotation.random().as_matrix()

                cloud_path = cloud_paths[data_num]
                label_path = label_paths[data_num]

                np_clouds, tag_np_cloud = read_pickle(cloud_path)
                tag_np_cloud = ((rand_R @ tag_np_cloud.T).T + rand_t)

                gt_labels = read_dict(label_path)
                torch_clouds = []
                is_tag = []
                target = []
                for cloud_ind in range(len(np_clouds)):
                    np_cloud = np_clouds[cloud_ind]
                    np_cloud = ((rand_R @ np_cloud.T).T + rand_t)

                    x_shift = np.random.uniform(-0.003, 0.003)
                    y_shift = np.random.uniform(-0.003, 0.003)
                    z_shift = np.random.uniform(-0.003, 0.003)
                    np_cloud = np_cloud + np.array([x_shift, y_shift, z_shift])
                    
                    pts_pct = np.random.uniform(0.75, 1.0)
                    rand_pts_inds = np.random.choice(np_cloud.shape[0], 
                                                     size=int(pts_pct*np_cloud.shape[0]),
                                                     replace=False)
                    np_cloud = np_cloud[rand_pts_inds]

                    gt_val = gt_labels[cloud_ind][-1] * 1.0
                    if np_cloud.shape[0] < opt.min_area:
                        continue
                    
                    #rand drop
                    #TODO if too few fg fruitlets don't drop
                    if gt_val > 0:
                        prob_drop = np.random.uniform(0, 0.3)
                    else:
                        prob_drop = np.random.uniform(0.2, 0.6)

                    rand_var = np.random.uniform()

                    if rand_var < prob_drop:
                        continue

                    torch_cloud = torch.from_numpy(np_cloud).float()
                    torch_cloud = torch_cloud.unsqueeze(0)
                    torch_cloud = torch_cloud.permute(0, 2, 1)
                    torch_cloud = torch_cloud.to(opt.device)
                    torch_clouds.append(torch_cloud)

                    is_tag.append(0.0)
                    target.append(gt_val)

                #TODO make this common code
                torch_tag_cloud = torch.from_numpy(tag_np_cloud).float()
                torch_tag_cloud = torch_tag_cloud.unsqueeze(0)
                torch_tag_cloud = torch_tag_cloud.permute(0, 2, 1)
                torch_tag_cloud = torch_tag_cloud.to(opt.device)
                torch_clouds.append(torch_tag_cloud)
                is_tag.append(1.0)
                target.append(False)

                ###shuffle
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

                fg_bg_loss = bce_loss_fn(pred[is_tag <= 0], target[is_tag <= 0])
                loss.append(fg_bg_loss)
                
        
            loss = torch.mean(torch.stack((loss)))
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
        
        scheduler.step()
        epoch_loss = np.mean(losses)
        print('loss for epoch', epoch, 'is: ', epoch_loss)

        if ((epoch + 1) % opt.log_steps) == 0:
            save_checkpoint(epoch + 1, opt.checkpoint_dir, classifier)
            loss_array.append(epoch_loss)
            plot_epochs.append(epoch + 1)
            plot_and_save_loss(opt.checkpoint_dir, plot_epochs, loss_array, True)

            #now evaluate
            classifier.eval()
            with torch.no_grad():
                precisions = []
                recalls = []
                f1_scores = []
                val_losses = []
                for _, data in enumerate(val_dataloader):
                    cloud_paths, label_paths = data
                    cloud_path = cloud_paths[0]
                    label_path = label_paths[0]
                    np_clouds, tag_np_cloud = read_pickle(cloud_path)
                    gt_labels = read_dict(label_path)
                    torch_clouds = []
                    is_tag = []
                    target = []
                    for cloud_ind in range(len(np_clouds)):
                        np_cloud = np_clouds[cloud_ind]
                        gt_val = gt_labels[cloud_ind][-1] * 1.0
                        if np_cloud.shape[0] < opt.min_area:
                            continue

                        torch_cloud = torch.from_numpy(np_cloud).float()
                        torch_cloud = torch_cloud.unsqueeze(0)
                        torch_cloud = torch_cloud.permute(0, 2, 1)
                        torch_cloud = torch_cloud.to(opt.device)
                        torch_clouds.append(torch_cloud)

                        is_tag.append(-1.0)
                        target.append(gt_val)

                    #TODO make this common code
                    torch_tag_cloud = torch.from_numpy(tag_np_cloud).float()
                    torch_tag_cloud = torch_tag_cloud.unsqueeze(0)
                    torch_tag_cloud = torch_tag_cloud.permute(0, 2, 1)
                    torch_tag_cloud = torch_tag_cloud.to(opt.device)
                    torch_clouds.append(torch_tag_cloud)
                    is_tag.append(1.0)
                    target.append(False)

                    is_tag = torch.as_tensor(is_tag, dtype=torch.float, device=opt.device)
                    target = torch.as_tensor(target, dtype=torch.float, device=opt.device)
                    pred = classifier(torch_clouds, is_tag).flatten()

                    pred = nn.functional.sigmoid(pred[is_tag <= 0])
                    target = target[is_tag <= 0]

                    val_loss = bce_loss_fn(pred, target)

                    pred = torch.where(pred >= opt.conf_thresh, 1.0, 0.0)

                    true_positive = torch.sum(pred*target)
                    false_positive = torch.sum(pred*(1-target))
                    false_negative = torch.sum((1-pred)*target)

                    if true_positive == 0:
                        precision = 0
                        recall = 0

                        precisions.append(precision)
                        recalls.append(recall)
                    else:
                        precision = true_positive / (true_positive + false_positive)
                        recall = true_positive / (true_positive + false_negative)

                        precisions.append(precision.item())
                        recalls.append(recall.item())

                    f1_numerator = 2*precision*recall
                    if f1_numerator == 0:
                        f1_score = 0
                        f1_scores.append(f1_score)
                    else:
                        f1_score = f1_numerator / (precision + recall)
                        f1_scores.append(f1_score.item())

                    
                    val_losses.append(val_loss.item())

            classifier.train()

            precision_array.append(np.mean(precisions))
            recall_array.append(np.mean(recalls))
            f1_array.append(np.mean(f1_scores))
            val_loss_array.append(np.mean(val_losses))

            plot_and_save_scores(opt.checkpoint_dir, plot_epochs, precision_array,
                                 recall_array, f1_array)
            plot_and_save_loss(opt.checkpoint_dir, plot_epochs, val_loss_array, False)

    save_checkpoint(opt.num_epochs, opt.checkpoint_dir, classifier)


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--cloud_dir', required=True)
    parser.add_argument('--label_dir', required=True)
    parser.add_argument('--val_dir', required=True)

    parser.add_argument('--checkpoint_dir', default='./checkpoints')

    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--min_area', type=int, default=50)
    parser.add_argument('--log_steps', type=int, default=10)

    parser.add_argument('--pos_weight', type=float, default=3.0)
    parser.add_argument('--conf_thresh', type=int, default=0.5)

    parser.add_argument('--shuffle', action='store_false')
    parser.add_argument('--device', default='cuda')
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    opt = parse_args()

    train(opt)