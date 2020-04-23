import os
import numpy as np
import cv2
from tqdm  import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from Model import VGG
from sklearn.model_selection import train_test_split

OUTPUT_PATH = "output/model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #GPUを使用

class Trainer():
    def __init__(self):
        self.model = VGG(9).to(device) #モデルの定義
        self.criterion = nn.CrossEntropyLoss().to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)

        #学習画像と正解を学習用とテスト用に分ける
        train_image, test_image, train_class, test_class = train_test_split(np.load('anime_face_image.npy'), np.load('anime_face_class.npy'), test_size=0.1)
        #data_loaderを作成
        train_data = torch.utils.data.TensorDataset(torch.Tensor(train_image), torch.Tensor(train_class))
        self.train_loader = torch.utils.data.DataLoader(train_data, batch_size=45, shuffle=True)
        test_data = torch.utils.data.TensorDataset(torch.Tensor(test_image), torch.Tensor(test_class))
        self.test_loader = torch.utils.data.DataLoader(test_data, batch_size=45, shuffle=True)

    def train(self,epoch):
        self.model.train()
        for batch_idx, (image, label) in enumerate(self.train_loader,1):
            label = label.type(torch.LongTensor)
            image, label = image.to(device), label.to(device)
            self.optimizer.zero_grad()
            output = self.model(image)
            loss = self.criterion(output, label)
            loss.backward()
            self.optimizer.step()

            tqdm.write('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(image), len(self.train_loader.dataset),
                100. * batch_idx / len(self.train_loader), loss.item()))

    def test(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        for (image, label) in self.test_loader:
            label = label.type(torch.LongTensor)
            image, label = image.to(device), label.to(device)
            output = self.model(image)
            test_loss += self.criterion(output, label)
            _, pred = torch.max(output.data,1)
            correct += (pred == label).sum().item()

        test_loss /= len(self.test_loader.dataset)
        tqdm.write('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(self.test_loader.dataset),
            100. * correct / len(self.test_loader.dataset)))

    def saveModel(self):
        torch.save(self.model.state_dict(), OUTPUT_PATH)

if __name__ == '__main__':
    trainer = Trainer()
    try:
        os.makedirs("output/")
    except:
        pass

    for epoch in tqdm(range(5)):
        trainer.train(epoch+1)
        trainer.test()

    trainer.saveModel()