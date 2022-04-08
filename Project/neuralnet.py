import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import joblib

# load label binarizer
print('Loading label binarizer...')
lb = joblib.load(r'C:\Users\Shreya Basu\Workspace\ASL-Translator\Project\outputs\lb.pkl')

class NeuralNet(nn.Module):
  def __init__(self, loss_fn, lrate):
    super(NeuralNet, self).__init__()
    self.loss_fn = loss_fn
    self.conv1 = nn.Conv2d(3, 16, 5)
    self.conv2 = nn.Conv2d(16, 32, 5)
    self.conv3 = nn.Conv2d(32, 64, 3)
    self.conv4 = nn.Conv2d(64, 128, 5)
    self.fc1 = nn.Linear(128, 256)
    self.fc2 = nn.Linear(256, len(lb.classes_))
    self.pool = nn.MaxPool2d(2, 2)
    self.optimizer = optim.Adam(self.parameters(), lr=lrate)
  def forward(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = self.pool(F.relu(self.conv3(x)))
    x = self.pool(F.relu(self.conv4(x)))
    bs, _, _, _ = x.shape
    x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return x