import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import os
import cv2
import matplotlib.pyplot as plt
import pandas as pd
# from tensorflow import keras
import glob
import keras
from keras.layers import Dropout
# from keras.layers.normalization import BatchNormalization
from sklearn.preprocessing import StandardScaler
import seaborn as sn

# 读取图像
def read_img(path):
    pic_size = 16               #TODO 设置初始化为 64 ，依据需求更改 图像像素大小一致
    cate = [path + x for x in os.listdir(path) if os.path.isdir(path + x)]
    imgs = []
    labels = []
    fpath = []
    for idx, folder in enumerate(cate):
        # 遍历整个目录判断每个文件是不是符合
        for im in glob.glob(folder + '/*.jpg'):
            # print('reading the images:%s' % (im))
            img = cv2.imread(im)  # 调用opencv库读取像素点
            img = cv2.resize(img, (pic_size, pic_size))
            imgs.append(img)            # 图像数据
            labels.append(idx)          # 图像类标
            fpath.append(path + im)  # 图像路径名
            # print(path+im, idx)

    return np.asarray(fpath, np.string_), np.asarray(imgs, np.float32), np.asarray(labels, np.int32)

#绘制训练过程损失函数和准确率的变化图
def plot_learning(history):
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.title('训练过程损失函数和准确率的变化图')
    plt.show()

# 绘制混淆矩阵
def plot_confusion_matrix(cm, title='混淆矩阵热力图', cmap=plt.cm.binary):
    # plt.figure(figsize=(11, 11))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(target_names)))
    plt.xticks(xlocations, target_names, rotation=0)
    plt.yticks(xlocations, target_names)
    plt.ylabel('正确标签')
    plt.xlabel('预测标签')
    plt.show()

# 设置字体
plt.rcParams['font.sans-serif'] = ['SimHei']
# 字符显示
plt.rcParams['axes.unicode_minus'] = False
# 定义图片路径
path = 'picture/Waterfall_Input/'
# 定义图片尺寸
pic_size = 16
# 读取图像
fpaths, data, label = read_img(path)
print(data.shape)  # (4224, 256, 256, 3) or (4224, 16, 16, 3)
# print(label.shape)
num_classes = len(set(label))      # 计算有多少类图片  # 按文件夹分7类

# 定义随机种子，将数据进行打乱处理
np.random.seed(116)
np.random.shuffle(data)
np.random.seed(116)
np.random.shuffle(label)

# 以下划分训练集，测试集和验证集
x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=1, stratify=label)
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.25, random_state=1, stratify=y_train)

# 定义对应的实际类别名称
target_names = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6']

# 数据归一化处理
# 1 创建一个StandardScaler对象，该对象用于标准化数据。
# 2 对StandardScaler进行拟合，计算出均值和标准差，并对训练集进行标准化。
# 3 这一步将图像数据的数据类型转换为float32，并将其形状从原始的(-1, pic_size, pic_size, 3)调整为(-1, 1)，以便StandardScaler能够处理。
# 4 最后，将标准化后的数据重新调整为原始图像数据的形状。这里的pic_size应该是之前定义的图像大小。
scaler = StandardScaler()
x_train_scaler = scaler.fit_transform(
    x_train.astype(np.float32).reshape(-1, 1)).reshape(-1, pic_size, pic_size, 3)       #TODO 原为 (-1, 64, 64, 3) 改为 pic_size
x_valid_scaler = scaler.fit_transform(
    x_valid.astype(np.float32).reshape(-1, 1)).reshape(-1, pic_size, pic_size, 3)
x_test_scaler = scaler.fit_transform(
    x_test.astype(np.float32).reshape(-1, 1)).reshape(-1, pic_size, pic_size, 3)

# 模型构建
model = keras.models.Sequential()
model.add(keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu', input_shape=(pic_size, pic_size, 3)))     #TODO 修改为 pic_size
model.add(keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))      #TODO 修改为 pic_size
model.add(keras.layers.MaxPool2D(pool_size=2))
model.add(Dropout(0.5))
model.add(keras.layers.Conv2D(filters=256,kernel_size=3,padding='same',activation='relu'))
model.add(keras.layers.Conv2D(filters=256,kernel_size=3,padding='same',activation='relu'))
model.add(keras.layers.MaxPool2D(pool_size=2))
model.add(Dropout(0.5))
model.add(keras.layers.Conv2D(filters=128,kernel_size=3,padding='same',activation='relu'))
model.add(keras.layers.Conv2D(filters=128,kernel_size=3,padding='same',activation='relu'))
model.add(keras.layers.MaxPool2D(pool_size=2))
model.add(Dropout(0.5))
model.add(keras.layers.Flatten())
#TODO 此处加入文本部分提取出的特征 进行拼接
model.add(keras.layers.Dense(64, activation='relu'))
# model.add(Dropout(0.25))  #随机退出率
model.add(keras.layers.Dense(10, activation="softmax"))
# 设计损失函数 优化器等参数
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="Adam", metrics=["accuracy"],)
# 显示模型的架构摘要
model.summary()

# TODO 模型训练 设计轮次
history = model.fit(x_train_scaler, y_train, epochs=100, validation_data=(x_valid_scaler, y_valid))
# TODO 此处定义了迭代次数 更新便利

# 画出模型在训练过程中的损失函数和准确率的变化
plot_learning(history)

# 模型预测
predict = model.predict(x_test_scaler)
# predict=model.evaluate(x_val_scaler,y_val)
predict = np.argmax(predict, axis=1)
# 输出测试集的精确率、召回率和F1测度等指标
print(classification_report(y_test, predict, target_names=target_names))

# 绘制混淆矩阵黑白
conmatrix = confusion_matrix(y_true=y_test, y_pred=predict)
plot_confusion_matrix(conmatrix)
print(conmatrix)

# # 绘制混淆矩阵彩色
# df_cm = pd.DataFrame(conmatrix,
#                      index=target_names,
#                      columns=target_names)
# plt.figure(figsize=(11, 11))
# # plt.yticks(rotation=90)
# plt.title('混淆矩阵对应正误个数图')
# sn.heatmap(df_cm, annot=True, cmap="BuPu")
# plt.show()
