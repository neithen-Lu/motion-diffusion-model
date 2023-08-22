import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import torchvision.transforms as TF
import numpy as np
import os
import pathlib
import random
from PIL import Image

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--batch_size', type=int, default=50,
                    help='Batch size to use')
parser.add_argument('--device', type=str, default=None,
                    help='Device to use. Like cuda, cuda:0 or cpu')
parser.add_argument('--num-workers', type=int,
                    help=('Number of processes to use for data loading. '
                          'Defaults to `min(8, num_cpus)`'))
parser.add_argument('--dims', type=int, default=2048,
                    help=('Dimensionality of Inception features to use. '
                          'By default, uses pool3 features'))
parser.add_argument('--num_samples', type=int, default=40000,
                    help='number of samples to calculate the FID score')
parser.add_argument('path', type=str, nargs=2,
                    help=('Paths to the generated images or '
                          'to .npz statistic files'))

IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm',
                    'tif', 'tiff', 'webp'}

class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, files, transforms=None):
        self.files = files
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        img = Image.open(path).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        return img

class NPZDataset(torch.utils.data.Dataset):
    def __init__(self, data_npz, transforms=None):
        self.transforms = transforms
        self.data = data_npz
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, i):
        img = self.data[i]
        if self.transforms is not None:
            img = self.transforms(img)

        return img

def get_data(path, num_workers, batch_size, num_samples=40000):
    if path.endswith('.npz'):
        x = np.load(path)
        files = x[x.files[0]]
        dataset = NPZDataset(files, transforms=TF.ToTensor())
    else:
        path = pathlib.Path(path)
        files = sorted([file for ext in IMAGE_EXTENSIONS
                       for file in path.glob('*.{}'.format(ext))])
        if len(files) > num_samples:
            files = sorted(random.sample(files, num_samples))
        dataset = ImagePathDataset(files, transforms=TF.ToTensor())
    dataloader = torch.utils.data.DataLoader(dataset,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            drop_last=False,
                                            num_workers=num_workers)
    return dataloader

def calculate_fid_kid(paths, num_workers, device, num_samples, batch_size):
    # get_data
    img_dist1_loader = get_data(paths[0], num_workers, batch_size, num_samples)
    img_dist2_loader = get_data(paths[1], num_workers, batch_size, num_samples)

    fid = FrechetInceptionDistance(normalize=True).to(device)
    kid = KernelInceptionDistance(normalize=True).to(device)

    for _, img_dist1 in enumerate(img_dist1_loader):
        fid.update(img_dist1.to(device), real=True)
        kid.update(img_dist1.to(device), real=True)

    for _, img_dist2 in enumerate(img_dist2_loader):
        fid.update(img_dist2.to(device), real=False)
        kid.update(img_dist2.to(device), real=False)

    fid = fid.compute()
    kid_mean, kid_sd = kid.compute()
    print('FID:', fid.item())
    print('KID: {} (mean); {} (sd)'.format(kid_mean.item(), kid_sd.item()))

def main():
    args = parser.parse_args()

    if args.device is None:
        device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    else:
        device = torch.device(args.device)

    if args.num_workers is None:
        try:
            num_cpus = len(os.sched_getaffinity(0))
        except AttributeError:
            # os.sched_getaffinity is not available under Windows, use
            # os.cpu_count instead (which may not return the *available* number
            # of CPUs).
            num_cpus = os.cpu_count()

        num_workers = min(num_cpus, 8) if num_cpus is not None else 0
    else:
        num_workers = args.num_workers
    
    calculate_fid_kid(args.path, num_workers, device, args.num_samples, args.batch_size)
    

if __name__ == '__main__':
    main()
    


