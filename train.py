from __future__ import print_function, division

import time
import os
from torchvision import datasets, models, transforms
import copy
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch
from torch.optim import lr_scheduler
import cv2
from utils import imshow, labelsToText
from data_iterator import get_dataloaders

data_dir = '/media/arshdees/70DBF1CB317140CE/Pex-ML-Challenge'
dataloaders, dataset_sizes, class_names = get_dataloaders(data_dir)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

'''
# Get a batch of training data using pytorch's dataloader object
inputs, classes = next(iter(dataloaders['train']))

## ! Uncommnet to visualize data with labels
out = torchvision.utils.make_grid(inputs[:4])
labelsMap = labelsToText()
imshow(out, title=[labelsMap[int(x)] for x in classes[0:4]])
'''

def train_model(model, criterion, optimizer, lr_scheduler, num_epochs=5):
    start = time.time()

    #deepcopy needed for references
    best_model = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    train_losses = []
    train_acc = []

    dev_losses = []
    dev_acc = []

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'dev']:
            if phase == 'train':
                lr_scheduler.step()
                #Toggle 'train' mode for model.
                model.train()
            else:
                #Toggle 'eval' mode for model
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # reset gradients
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # accumulate
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase == 'train':
                train_losses.append(epoch_loss)
                train_acc.append(epoch_acc)
            elif phase == 'dev':
                dev_losses.append(epoch_loss)
                dev_acc.append(epoch_acc)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            #keep updating best model
            if phase == 'dev' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best dev Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model)
    return model, train_losses, train_acc, dev_losses, dev_acc

#Transfer Learning: Using our classification layer instaed of original ResNet's

model_ft = models.resnet18(pretrained=True, )

#model_ft.fc = nn.Linear(model_ft.fc.in_features, 2)
model_ft.fc = nn.Sequential(nn.Linear(model_ft.fc.in_features, 512),
                         nn.ReLU(),
                         nn.Dropout(0.2),
                         nn.Linear(512, 32),
                         nn.ReLU(),
                         nn.Dropout(0.2),
                         nn.Linear(32, 2)
                         )

print(model_ft)
model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

#optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.003)
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.003, momentum=0.7)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, 5, gamma=0.1)

model, train_losses, train_acc, dev_losses, dev_acc = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=10)

torch.save(model, 'pex-challenge-model.pyt')

plt.ioff()
plt.plot(train_losses, label='Training loss')
plt.plot(dev_losses, label='Validation loss')
plt.legend(frameon=False)
plt.show()

plt.plot(train_acc, label='Training Accuracy')
plt.plot(dev_acc, label='Validation Accuracy')
plt.legend(frameon=False)
plt.show()


