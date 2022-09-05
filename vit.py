import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import japanize_matplotlib

#gpu確認
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#学習済みViTモデルのロード
model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)

#パラメータ固定
for param in model.parameters():
    param.requires_grad = False

#最終層の設定
model.heads[0] = nn.Linear(768, 2)

model.to(device)

#データ準備
root = 'C:/Users/ofami/python/vit/dog_wolf'
train_dir = os.path.join(root, 'train')
test_dir = os.path.join(root, 'test')
classes = ['dog', 'wolf']

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5), 
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)   
])
# 検証データ用 : 正規化のみ実施
test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)
])

train_data = datasets.ImageFolder(train_dir, transform=train_transform)
test_data = datasets.ImageFolder(test_dir, transform=test_transform)

train_loader = DataLoader(train_data, batch_size=5, shuffle=True)
test_loader = DataLoader(test_data, batch_size=5, shuffle=True)

#ハイパーパラメータ
lr = 3e-5
epochs = 200

#損失関数と評価関数
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

history = np.zeros((0,5))

for epoch in range(epochs):
    train_loss = 0
    train_acc = 0
    val_loss = 0
    val_acc = 0

    #訓練フェーズ
    model.train()
    count = 0

    for inputs, labels in tqdm(train_loader):
        count += len(labels)
        inputs = inputs.to(device)
        labels = labels.to(device)

        #勾配の初期化
        optimizer.zero_grad()

        #予測計算
        outputs = model(inputs)

        #損失計算
        loss = criterion(outputs, labels)
        train_loss += loss.item()

        #勾配計算
        loss.backward()

        #パラメータ更新
        optimizer.step()

        #予測値算出
        predicted = torch.max(outputs, 1)[1]

        #正解数算出
        train_acc += (predicted == labels).sum()

        #損失と精度の計算
        avg_train_loss = train_loss / count
        avg_train_acc = train_acc / count
    
    #予測フェーズ
    model.eval()
    count = 0

    for inputs, labels in test_loader:
        count += len(labels)

        inputs = inputs.to(device)
        labels = labels.to(device)

        # 予測計算
        outputs = model(inputs)

        # 損失計算
        loss = criterion(outputs, labels)
        val_loss += loss.item()

        #予測値算出
        predicted = torch.max(outputs, 1)[1]

        #正解件数算出
        val_acc += (predicted == labels).sum()

        # 損失と精度の計算
        avg_val_loss = val_loss / count
        avg_val_acc = val_acc / count

    print(f'epoch:{epoch}, loss: {avg_train_loss:.5f} acc: {avg_train_acc:.5f} val_loss: {avg_val_loss:.5f}, val_acc: {avg_val_acc:.5f}')

    item = np.array([epoch + 1, avg_train_loss, avg_train_acc, avg_val_loss, avg_val_acc])
    history = np.vstack((history, item))

#学習結果の確認
for images, labels in test_loader:
    break

n = len(images)

inputs = images.to(device)
labels = labels.to(device)

outputs = model(inputs)
predicted = torch.max(outputs, 1)[1]

plt.figure(figsize=(20,15))
for i in range(n):
    ax = plt.subplot(5, 10, i + 1)
    label_name = classes[labels[i]]
    
    predicted_name = classes[predicted[i]]
    # 正解かどうかで色分けをする
    if label_name == predicted_name:
        c = 'k'
    else:
        c = 'b'
    ax.set_title(label_name + ':' + predicted_name, c=c, fontsize=20)
    # TensorをNumPyに変換
    image_np = images[i].numpy().copy()
    # 軸の順番変更 (channel, row, column) -> (row, column, channel)
    img = np.transpose(image_np, (1, 2, 0))
    # 値の範囲を[-1, 1] -> [0, 1]に戻す
    img = (img + 1)/2
    # 結果表示
    plt.imshow(img)
    ax.set_axis_off()
plt.show()

#損失と精度の確認
print(f'初期状態: 損失: {history[0,3]:.5f} 精度: {history[0,4]:.5f}') 
print(f'最終状態: 損失: {history[-1,3]:.5f} 精度: {history[-1,4]:.5f}' )

num_epochs = len(history)
unit = num_epochs / 10

# 学習曲線の表示 (損失)
plt.figure(figsize=(9,8))
plt.plot(history[:,0], history[:,1], 'b', label='訓練')
plt.plot(history[:,0], history[:,3], 'k', label='検証')
plt.xticks(np.arange(0,num_epochs+1, unit))
plt.xlabel('繰り返し回数')
plt.ylabel('損失')
plt.title('学習曲線(損失)')
plt.legend()
plt.show()

# 学習曲線の表示 (精度)
plt.figure(figsize=(9,8))
plt.plot(history[:,0], history[:,2], 'b', label='訓練')
plt.plot(history[:,0], history[:,4], 'k', label='検証')
plt.xticks(np.arange(0,num_epochs+1,unit))
plt.xlabel('繰り返し回数')
plt.ylabel('精度')
plt.title('学習曲線(精度)')
plt.legend()
plt.show()