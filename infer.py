import argparse
import os
import torch
import cv2
from data.dataloader import get_data_loader
from models.fb_bg_classifier import FgBgClassifier
from utils.utils import load_checkpoint, read_pickle, read_dict

def infer(opt):
    dataloader = get_data_loader(opt.cloud_dir, opt.label_dir, 
                                 1, False)
    classifier = FgBgClassifier().to(opt.device)

    load_checkpoint(opt.checkpoint_epoch, opt.checkpoint_dir, classifier)
    classifier.eval()

    with torch.no_grad():
        for _, data in enumerate(dataloader):
            cloud_paths, label_paths = data
            #TODO make this common
            for data_num in range(len(cloud_paths)):
                cloud_path = cloud_paths[data_num]
                label_path = label_paths[data_num]
                np_clouds, tag_np_cloud = read_pickle(cloud_path)
                gt_labels = read_dict(label_path)
                torch_clouds = []
                is_tag = []
                target = []
                boxes = []
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
                    boxes.append(gt_labels[cloud_ind][0:4])

                #TODO shuffle tag ind
                #TOODO make this common code
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

                pred = pred[is_tag <= 0]

                image_path = os.path.join(opt.image_dir, 
                                          os.path.basename(label_path).replace('.json', 
                                                                               '.png'))                
                im = cv2.imread(image_path)
                for node_ind in range(pred.shape[0]):
                    if pred[node_ind].item() > 0:
                        color = [0, 0, 255]
                    else:
                        color = [0, 255, 0]

                    x0, y0, x1, y1 = boxes[node_ind]

                    cv2.rectangle(im, (int(x0), int(y0)), (int(x1), int(y1)), color, vis_thickness)

                output_path = os.path.join(opt.vis_dir, 
                                           os.path.basename(image_path))
                
                cv2.imwrite(output_path, im)


vis_thickness = 2
default_image_dir = '/home/frc-ag-3/harry_ws/fruitlet_2023/labelling/inhand/fg_bg_images'
def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--cloud_dir', required=True)
    parser.add_argument('--label_dir', required=True)

    parser.add_argument('--checkpoint_dir', default='./checkpoints')
    parser.add_argument('--checkpoint_epoch', type=int, required=True)
    parser.add_argument('--vis_dir', default='./vis')
    parser.add_argument('--image_dir', default=default_image_dir)

    parser.add_argument('--min_area', type=int, default=50)

    parser.add_argument('--device', default='cuda')
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    opt = parse_args()

    infer(opt)