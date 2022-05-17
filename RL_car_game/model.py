import torch
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F 
import os

class Linear_QNet(nn.Module):  
    def __init__(self): 
        super().__init__() # lets say it takes 256*256*3 input
        self.conv1 = nn.Conv2d(3, 32, 8, 2) # input channels, channels, kernel size, strides
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 4)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 64, 2)
        self.relu3 = nn.ReLU(inplace=True)
        self.maxpool3 = nn.MaxPool2d(2, 2)
        self.linear1 = nn.Linear(12544, 1024)
        self.relu4 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(1024, 512)
        self.relu5 = nn.ReLU(inplace=True)
        self.linear3 = nn.Linear(512, 8)
        self.softmax1 = nn.Softmax(-1)

    def forward(self, x): 
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.maxpool1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.maxpool2(out)
        out = self.conv3(out)
        out = self.relu3(out)
        out = self.maxpool3(out)
        self.dense_input = out.shape # use this to fill input to linear layer
        out = out.view(out.size(0), -1)
        out = self.linear1(out)
        out = self.relu4(out)
        out = self.linear2(out)
        out = self.relu5(out)
        out = self.linear3(out)
        out = self.softmax1(out)
        return out 

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr = self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # (n, x)

        if len(action.shape) == 1:
            # (1, x)
            state = state.unsqueeze(0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 1: predicted Q values with current state
        pred = self.model(state)

        target = pred.clone() #
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action).item()] = Q_new

        # 2: r + y * max(next_predicted Q value)   // y is gamma
        # pred.clone()
        # preds[argmax(action)] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()


if __name__ == '__main__':
    net = Linear_QNet()
    x = torch.randn(1, 3, 256, 256)
    out = net(x)
    print(out)
    print(out.shape)
    pytorch_total_params = sum(p.numel() for p in net.parameters())
    print(pytorch_total_params)
