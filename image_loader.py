import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch

def get_loader(root_folder,
               batch_size,
               num_workers=4):
    dataset = dset.ImageFolder(root=root_folder,
                               transform=transforms.Compose([
                                   transforms.CenterCrop(160),
                                   transforms.Scale(64),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=num_workers)
    return dataloader

if __name__=='__main__':
    dataloader = get_loader("CelebA", 16, 4)
    for i, data in enumerate(dataloader):
        print(data[0])
        break
