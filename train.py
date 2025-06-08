import torch  # 导入PyTorch库
import torch.nn as nn  # 导入PyTorch的神经网络模块
from PIL import Image  # 导入PIL图像处理库中的Image模块
import torchvision.transforms as transforms  # 导入PyTorch视觉处理库中的transforms模块
from torch.nn import functional as F  # 导入PyTorch的函数式模块
from torch.utils import data  # 导入PyTorch的数据工具模块
from torch.utils.data import Dataset  # 导入PyTorch的数据集类
import torchvision  # 导入PyTorch的视觉库
import os  # 导入Python的os模块
from torch.utils.data import random_split  # 导入PyTorch的随机分割数据集模块
from torch.utils.tensorboard import SummaryWriter  # 导入PyTorch的TensorBoard模块
import shutil  # 导入shutil模块用于删除目录

# 清理旧的日志文件
log_dir = "logs"
if os.path.exists(log_dir):
    shutil.rmtree(log_dir)
writer = SummaryWriter(log_dir)  # 创建一个新的TensorBoard记录器
cur_path = os.path.dirname(__file__)  # 获取当前文件所在的目录路径
data_dir = os.path.join(cur_path, 'data')  # 数据集所在的目录路径

# 配置设备，如果CUDA可用，则使用GPU，否则使用CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
min_loss = float('inf')  # 初始化最小损失值为正无穷
# 设置超参数
num_epochs = 15
num_classes = 4
batch_size = 128
learning_rate = 0.0015
model_path = os.path.join(cur_path, 'model.pth')  # 模型保存路径
n_epochs_stop = 3  # 如果连续多少个epoch没有改善，则停止训练
best_val_loss = float("Inf")  # 初始化最佳验证损失值为正无穷

class EmojiDataset(Dataset):  # 自定义数据集类
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir  # 数据集根目录
        self.transform = transform  # 数据预处理操作
        self.classes = os.listdir(root_dir)  # 类别列表
        self.images = []
        # 遍历数据集目录，获取图像路径及其对应的标签
        for class_ in self.classes:
            class_dir = os.path.join(root_dir, class_)
            for image_name in os.listdir(class_dir):
                self.images.append((os.path.join(class_dir, image_name), self.classes.index(class_)))

    def __len__(self):
        return len(self.images)  # 返回数据集的长度

    def __getitem__(self, idx):
        image_path, label = self.images[idx]  # 获取图像路径和标签
        image = Image.open(image_path)  # 使用PIL库打开图像
        if self.transform:  # 如果定义了数据预处理操作
            image = self.transform(image)  # 对图像进行预处理
        return image, label  # 返回处理后的图像和标签

# 定义数据预处理操作
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

# 创建数据集对象
dataset = EmojiDataset(data_dir)
# 确定数据集分割的长度
lengths = [int(len(dataset) * 0.8), len(dataset) - int(len(dataset) * 0.8)]
# 创建训练集和测试集的分割
train_dataset, test_dataset = random_split(dataset, lengths)
train_dataset.dataset.transform = data_transform['train']  # 设置训练集的数据预处理
test_dataset.dataset.transform = data_transform['test']  # 设置测试集的数据预处理

# 创建数据加载器
train_loader = data.DataLoader(dataset=train_dataset,
                               batch_size=batch_size,
                               shuffle=True)
test_loader = data.DataLoader(dataset=test_dataset,
                              batch_size=batch_size,
                              shuffle=False)

# 定义残差模块
class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)

# 定义残差块
def block(input_channels, num_channels, num_residuals, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            output = Residual(input_channels, num_channels, use_1x1conv=True, strides=2)
            blk.append(output)
        else:
            output = Residual(num_channels, num_channels)
            blk.append(output)
    return nn.Sequential(*blk)

# 定义一个函数，用于输出每一层的形状
def shape(net):
    X = torch.rand(size=(1, 3, 224, 224))
    for layer in net:
        X = layer(X)
        print(layer.__class__.__name__, 'output shape: ', X.shape)

# 构建ResNet模型
b1 = nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
b2 = block(64, 64, 2, first_block=True)
b3 = block(64, 128, 2)
b4 = block(128, 256, 2)
b5 = block(256, 512, 2)
net = nn.Sequential(b1, b2, b3, b4, b5, nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(512, num_classes))

# 将网络结构可视化到TensorBoard
writer.add_graph(net, input_to_model=torch.rand(1, 3, 224, 224))
model = net.to(device)  # 将模型移到GPU上（如果可用）

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))  # 如果存在预训练模型，则加载模型参数

if __name__ == '__main__':
    print(model)  # 打印模型结构

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 记录超参数
    writer.add_hparams(
        {
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'num_epochs': num_epochs,
            'num_classes': num_classes
        },
        {}
    )

    # 训练模型
    total_step = len(train_loader)  # 总步数
    current_train_step = 1  # 当前训练步数
    current_test_step = 1  # 当前测试步数

    for epoch in range(num_epochs):
        model.train()  # 设置为训练模式
        running_loss = 0.0
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

            running_loss += loss.item()
            
            # 每20步记录一次训练损失
            if (i + 1) % 20 == 0:
                avg_loss = running_loss / 20
                writer.add_scalar("Training/Loss", avg_loss, current_train_step)
                running_loss = 0.0
                
                if loss < min_loss:
                    torch.save(model.state_dict(), model_path)
                    min_loss = loss
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
            
            current_train_step += 1

        # 测试模型
        model.eval()  # 切换到评估模式
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                batch_loss = criterion(outputs, labels)
                val_loss += batch_loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # 记录测试损失和准确率
                writer.add_scalar("Testing/Loss", batch_loss.item(), current_test_step)
                writer.add_scalar("Testing/Accuracy", 100 * (predicted == labels).sum().item() / labels.size(0),
                                current_test_step)
                current_test_step += 1

            # 记录每个epoch的平均测试损失和准确率
            avg_val_loss = val_loss / len(test_loader)
            accuracy = 100 * correct / total
            writer.add_scalar("Epoch/Validation_Loss", avg_val_loss, epoch)
            writer.add_scalar("Epoch/Accuracy", accuracy, epoch)
            
            print('Test Accuracy of the model on the test images: {:.2f}%'.format(accuracy))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve == n_epochs_stop:
            print('Early stopping!')
            break

writer.close()  # 关闭TensorBoard记录器
