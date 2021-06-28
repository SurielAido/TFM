import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data.sampler import SubsetRandomSampler

data_dir = '../core/dataset/train'


def load_split_train_test(datadir, valid_size=.2):
    train_transforms = transforms.Compose([transforms.Resize((56, 56)),
                                           transforms.ToTensor(),
                                           ])
    test_transforms = transforms.Compose([transforms.Resize((56, 56)),
                                          transforms.ToTensor(),
                                          ])
    train_data = datasets.ImageFolder(datadir,
                                      transform=train_transforms)
    test_data = datasets.ImageFolder(datadir,
                                     transform=test_transforms)
    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    np.random.shuffle(indices)

    train_idx, test_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    trainloader = torch.utils.data.DataLoader(train_data,
                                              sampler=train_sampler, batch_size=64)
    testloader = torch.utils.data.DataLoader(test_data,
                                             sampler=test_sampler, batch_size=64)
    return trainloader, testloader


trainloader, testloader = load_split_train_test(data_dir, .2)
print(trainloader.dataset.classes)

device = torch.device("cuda" if torch.cuda.is_available()
                      else "cpu")
model = models.resnet50(pretrained=True)
# print(model)

for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Sequential(nn.Linear(2048, 700),
                         nn.ReLU(),
                         nn.Dropout(p=0.2),
                         nn.Linear(700, 300),
                         nn.ReLU(),
                         nn.Dropout(p=0.2),
                         nn.Linear(300, 196),
                         nn.LogSoftmax(dim=1))
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.003)
model.to(device)

epochs = 100
steps = 100
running_loss = 0
print_every = 10
train_losses, test_losses = [], []
print("Vamos a entrenar")
for epoch in range(epochs):
    for inputs, labels in trainloader:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        steps += 1
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in testloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    test_loss += batch_loss.item()

                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            train_losses.append(running_loss / len(trainloader))
            test_losses.append(test_loss / len(testloader))
            end.record()

            # Waits for everything to finish running
            torch.cuda.synchronize()
            print(f"Epoch {epoch + 1}/{epochs}.. "
                  f"Train loss: {running_loss / print_every:.3f}.. "
                  f"Test loss: {test_loss / len(testloader):.3f}.. "
                  f"Test accuracy: {accuracy / len(testloader):.3f}.."
                  f"Time: {start.elapsed_time(end)}")
            running_loss = 0
            model.train()
torch.save(model, 'aerialmodel.pth')