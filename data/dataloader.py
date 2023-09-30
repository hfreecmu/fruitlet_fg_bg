import os
from torch.utils.data import DataLoader, Dataset

class FgBgDataset(Dataset):
    def __init__(self, cloud_dir, label_dir):
        self.paths = self.get_paths(cloud_dir, label_dir)

    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        return self.paths[idx]

    def get_paths(self, cloud_dir, label_dir):
        paths = []
        for filename in os.listdir(cloud_dir):
            if not filename.endswith('.pkl'):
                continue
            
            cloud_path = os.path.join(cloud_dir, filename)
            label_path = os.path.join(label_dir, filename.replace('.pkl', '.json'))

            if not os.path.exists(label_path):
                continue

            paths.append([cloud_path, label_path])

        return paths

def get_data_loader(cloud_dir, label_dir, batch_size, shuffle):
    dataset = FgBgDataset(cloud_dir, label_dir)
    dloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)

    return dloader