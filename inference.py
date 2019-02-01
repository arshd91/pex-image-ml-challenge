import torch
import cv2
import numpy as np
import sys
from data_iterator import get_transforms, get_dataloaders
from torchvision import datasets, models, transforms
import os
from utils import labelsToText
from PIL import Image
from torch.autograd import Variable
from utils import imshow
from torchvision import datasets, models, transforms

def get_random_images(num, datasets, data_dir, transform):
    data = datasets.ImageFolder(data_dir, transform=transform)
    classes = data.classes
    indices = list(range(len(data)))
    np.random.shuffle(indices)
    idx = indices[:num]
    from torch.utils.data.sampler import SubsetRandomSampler
    sampler = SubsetRandomSampler(idx)
    loader = torch.utils.data.DataLoader(data, sampler=sampler, batch_size=num)
    dataiter = iter(loader)
    images, labels = dataiter.next()
    return images, labels

def load_model(model_path):

    model = torch.load(model_path)
    return model

def predict_image(image, labelsMap, data_transforms):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    image_tensor = data_transforms['dev'](image)
    image_tensor = image_tensor.unsqueeze_(0)
    input = Variable(image_tensor)
    input = input.to(device)
    output = model(input)
    index = output.data.cpu().numpy().argmax()
    return labelsMap[index]

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('usage: python3 inference.py [model-path] [data_dir]')
        exit(0)

    model_path = sys.argv[1]
    data_dir = sys.argv[2]

    model = load_model(model_path)
    model.eval()
    transform = get_transforms()

    img = Image.open(data_dir)

    res = predict_image(img, labelsToText(), transform)
    print("Predicted Class: " + str(res))

    T2  = transforms.ToTensor()
    img = T2(img)
    imshow(img, res)
