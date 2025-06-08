import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
import os
from torch.utils.tensorboard import SummaryWriter

# 创建一个logs文件夹，writer写的文件都在该文件夹下
writer = SummaryWriter("logs")

cur_path = os.path.dirname(__file__)
data_dir = os.path.join(cur_path, 'data')

# 设备配置
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
min_loss = float('inf')

# 超参数
num_epochs = 50
num_classes = 4
batch_size = 128
learning_rate = 0.0005
model_path = os.path.join(cur_path, 'model.pth')
n_epochs_stop = 3  # 如果没有改善，停止的轮数
best_val_loss = float("inf")

class EmojiDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(root_dir)
        self.images = []
        for class_ in self.classes:
            class_dir = os.path.join(root_dir, class_)
            for image_name in os.listdir(class_dir):
                self.images.append((os.path.join(class_dir, image_name), self.classes.index(class_)))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path, label = self.images[idx]
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        return image, label

data_transform = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# 创建数据集
dataset = EmojiDataset(data_dir)
# 确定数据集的切分长度
lengths = [int(len(dataset)*0.8), len(dataset) - int(len(dataset)*0.8)]
# 创建训练集和验证集
train_dataset, test_dataset = random_split(dataset, lengths)
train_dataset.dataset.transform = data_transform['train']
test_dataset.dataset.transform = data_transform['test']

# 数据加载器
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True)

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=batch_size,
                         shuffle=False)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()        
        self.model1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),#32*224*224
            nn.BatchNorm2d(32),
            nn.MaxPool2d(7),#32*32*32
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),#32*32*32
            nn.BatchNorm2d(32),
            nn.MaxPool2d(4),#32*8*8
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1, 1),#64*14*14
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2)#64*4*4
        )
        self.fc = nn.Sequential(
            nn.Linear(64*4*4, 64),
            nn.Dropout(0.5),  # 添加Dropout，丢弃概率为0.5
            nn.Linear(64, 4)
        )
        
    def forward(self, x):
        x = self.model1(x)
        x = x.view(-1,64*4*4)  # 展平操作
        x = self.fc(x)
        return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
modeltest=CNN()
writer.add_graph(modeltest, input_to_model = torch.rand(1, 3, 224, 224))

model = CNN().to(device)  # 实例化模型并放到设备上
# writer.add_graph(model, input_to_model=torch.rand(1, 3, 224, 224))  # 添加模型结构图到tensorboard
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))
if __name__ == '__main__':
    print(model)
    print(f'模型有{count_parameters(model):,}个可训练参数.')

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,weight_decay=0.01)

    total_step = len(train_loader)
    current_train_step = 1
    current_test_step = 1

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            writer.add_scalar("train_loss", loss.item(), current_train_step)
            current_train_step += 1

            if (i + 1) % 20 == 0:
                if loss < min_loss:
                    torch.save(model.state_dict(), model_path)
                    min_loss = loss
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

        val_loss = 0
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                batch_loss = criterion(outputs, labels)
                writer.add_scalar("test_loss", batch_loss.item(), current_test_step)

                val_loss += batch_loss.item()
                _, predicted = torch.max(outputs.data, 1)
                writer.add_scalar("single_test_accuracy", 100 * (predicted == labels).sum().item() / labels.size(0), current_test_step)
                current_test_step += 1
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print('测试集准确率: {} %'.format(100 * correct / total))
            writer.add_scalar("total_test_accuracy", 100 * correct / total, epoch)

        if val_loss <  best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # if epochs_no_improve == n_epochs_stop:
        #     print('提前停止训练!')
        #     break

writer.close()
