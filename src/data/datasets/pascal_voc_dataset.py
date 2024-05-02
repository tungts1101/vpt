from torchvision import datasets
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from .transformations import (
    get_train_transformation,
    get_finetune_train_transformation,
    get_val_transformation,
    BarlowTwinTransform
)


class BarlowTwinPascalVOCDataset(Dataset):
    def __init__(self, data_dir, split="train"):
        self.root_dataset = datasets.VOCSegmentation(
            root=data_dir, year="2012", image_set=split, download=True
        )
        self.split = split
        if split == 'train':
            self.transform = BarlowTwinTransform()
        else:
            self.transform = get_val_transformation()

    def __len__(self):
        return len(self.root_dataset)

    def __getitem__(self, index):
        sample = self.root_dataset[index]
        return self.transform(sample)
    

class PascalVOCDataset(Dataset):
    def __init__(self, data_dir, split="train", transform=None):
        self.root_dataset = datasets.VOCSegmentation(
            root=data_dir, year="2012", image_set=split, download=True
        )
        self.transform = transform

    def __len__(self):
        return len(self.root_dataset)

    def __getitem__(self, index):
        sample = self.root_dataset[index]
        image, mask = sample
        if self.transform:
            image, mask = self.transform(sample)
        return image, mask

def get_train_dataset(cfg):
    return PascalVOCDataset(cfg.DATA.DATAPATH, transform=get_train_transformation())

def get_val_dataset(cfg):
    return PascalVOCDataset(cfg.DATA.DATAPATH, split='val', transform=get_val_transformation())

def get_train_dataloader(args):
    if args.train:
        dataset = PascalVOCDataset(args.data_dir, transform=get_train_transformation())
    elif args.finetune:
        dataset = PascalVOCDataset(
            args.data_dir, transform=get_finetune_train_transformation()
        )
    else:
        raise ValueError("Invalid training mode")

    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )


def get_val_dataloader(args):
    dataset = PascalVOCDataset(
        args.data_dir, split="val", transform=get_val_transformation()
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )

def get_gt_val_dataset(args):
    return PascalVOCDataset(args.data_dir, split='val', transform=None)