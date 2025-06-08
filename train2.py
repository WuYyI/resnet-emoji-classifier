import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
from torch.nn import functional as F
from torch.utils import data
from torch.utils.data import Dataset
import torchvision
import os
from torch.utils.data import random_split
from  torch.utils.tensorboard  import  SummaryWriter 

# 创建一个 TensorBoard 的 SummaryWriter 对象，用于记录日志
writer  =  SummaryWriter ( "logs" )  

# 获取当前文件路径和数据目录路径
cur_path=os.path.dirname(__file__)
data_dir=os.path.join(cur_path,'data')

# 设备配置（如果有 GPU 则使用 GPU，否则使用 CPU）
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 初始化变量
min_loss = float('inf')  # 最小损失
num_epochs = 15  # 训练周期数
num_classes = 4  # 数据集中的类别数
batch_size = 128  # 批处理大小
learning_rate = 0.0015  # 学习率
model_path=os.path.join(cur_path,'model.pth')  # 保存训练模型的路径
n_epochs_stop = 3  # 若连续多少周期损失未改善，则停止训练
best_val_loss = float("Inf")  # 最佳验证损失

# 数据集类的定义
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
        image= Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        return image, label
    
# 数据变换
data_transform={
    'train':transforms.Compose([ 
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])   
    ]),
    'test':transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])   
    ])
}

# 创建数据集
dataset = EmojiDataset(data_dir)

# 计算数据集划分的长度
lengths = [int(len(dataset)*0.8), len(dataset) - int(len(dataset)*0.8)]

# 创建训练和验证数据集
train_dataset, test_dataset = random_split(dataset, lengths)

# 对训练和测试数据集应用相应的变换
train_dataset.dataset.transform = data_transform['train'] 
test_dataset.dataset.transform = data_transform['test']

# 数据加载器
train_loader = data.DataLoader(dataset=train_dataset,
                               batch_size=batch_size,
                               shuffle=True)

test_loader = data.DataLoader(dataset=test_dataset,
                              batch_size=batch_size,
                              shuffle=False)

# Residual 模块的定义
class Residual(nn.Module):
    def __init__(self,input_channels,num_channels,use_1x1conv=False,strides=1):
        super().__init__()
        self.conv1=nn.Conv2d(input_channels,num_channels,kernel_size=3,padding=1,stride=strides)
        self.conv2=nn.Conv2d(num_channels,num_channels,kernel_size=3,padding=1)
        if use_1x1conv:
            self.conv3=nn.Conv2d(input_channels,num_channels,kernel_size=1,stride=strides)
        else:
            self.conv3=None
        self.bn1=nn.BatchNorm2d(num_channels)
        self.bn2=nn.BatchNorm2d(num_channels)
        self.relu=nn.ReLU(inplace=True)
    def forward(self,X):
        Y=F.relu(self.bn1(self.conv1(X)))
        Y=self.bn2(self.conv2(Y))
        if self.conv3:
            X=self.conv3(X)
        Y+=X
        return F.relu(Y)

# Residual 模块的堆叠
def block(input_channels, num_channels, num_residuals,first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            output=Residual(input_channels,num_channels, use_1x1conv=True, strides=2)
            blk.append(Residual(input_channels,num_channels, use_1x1conv=True, strides=2))
        else:
            output=Residual(num_channels,num_channels)
            blk.append(Residual(num_channels,num_channels))
        
    return nn.Sequential(*blk)

# 模型结构定义
b1= nn.Sequential(
            nn.Conv2d(3,64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(), nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
b2=block(64,64,2,first_block=True)
b3=block(64,128,2)

net=nn.Sequential(b1,b2,b3,nn.AdaptiveAvgPool2d((1,1)),nn.Flatten(),nn.Linear(128,num_classes))

# 将模型结构添加到 TensorBoard 中以绘制计算图
writer.add_graph(net,input_to_model = torch.rand(1,3,224,224))    
model = net.to(device)

# 计算模型参数数量的函数
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 如果已保存模型文件存在，则加载模型参数
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))

if __name__ == '__main__':
    print(model)
    print(f'模型有{count_parameters(model):,}个可训练参数.')

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 训练模型
    total_step = len(train_loader)
    current_train_step=1
    current_test_step=1

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss = criterion(outputs, labels)
            writer.add_scalar("train_loss", loss.item(), current_train_step)
            current_train_step=current_train_step+1          

            if (i + 1) % 20 == 0:
                if loss < min_loss:
                    torch.save(model.state_dict(), model_path)
                    min_loss = loss
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                    .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
        
        # 测试模型
        val_loss=0
        model.eval()  # 评估模式（batchnorm 使用移动平均/方差而不是小批量平均/方差）
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                batch_loss = criterion(outputs, labels)
                writer.add_scalar("test_loss", batch_loss.item(), current_test_step)
                
                # 累积损失
                val_loss += batch_loss.item()
                _, predicted = torch.max(outputs.data, 1)
                writer.add_scalar("single_test_accuracy", 100 * (predicted == labels).sum().item() / labels.size(0), current_test_step)
                current_test_step+=1
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))
            writer.add_scalar("total_test_accuracy", 100 * correct / total, epoch )

        if val_loss <  best_val_loss:
            # 若损失改善，则保存模型并重置计数器
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            # 若损失未改善，则增加计数器
            epochs_no_improve += 1  
        print(best_val_loss)    
        if epochs_no_improve == n_epochs_stop:
            print('Early stopping!')
            break 
writer.close()  # 关闭 TensorBoard 日志记录器
