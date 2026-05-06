import os
import torch
import h5py


class WSI_Dataset(torch.utils.data.Dataset):
    def __init__(self, folder, fixed_size=512):
        self.folder = folder
        self.fixed_size = fixed_size
        self.feat_files = []
        relative_ims = os.listdir(folder)
        for name in relative_ims:
            if name[-2:] == 'h5':
                self.feat_files.append(os.path.join(folder, name))


    def __getitem__(self, idx):
        file = self.feat_files[idx]
        with h5py.File(file, 'r') as h5:
            feats = torch.from_numpy(h5['feats'][:]).float()
            coords = torch.from_numpy(h5['coords'][:])
        

        target = int('positive' in file)
        target = torch.tensor(target).float()

        return feats, target, coords

    def __len__(self):
        return len(self.feat_files)
