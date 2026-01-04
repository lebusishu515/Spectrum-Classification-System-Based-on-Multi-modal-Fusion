import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import os
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import glob
import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense, concatenate
from keras.models import Model
from sklearn.preprocessing import StandardScaler
import seaborn as sn

#加入
def read_img(path):
    pic_size = 16
    cate = [path + x for x in os.listdir(path) if os.path.isdir(path + x)]
    imgs = []
    labels = []
    fpath = []
    for idx, folder in enumerate(cate):
        for im in glob.glob(folder + '/*.jpg'):
            img = cv2.imread(im)
            img = cv2.resize(img, (pic_size, pic_size))
            imgs.append(img)
            labels.append(idx)
            fpath.append(path + im)

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
plt.rcParams['axes.unicode_minus'] = False
path_x = 'picture/Waterfall_Input/'
pic_size = 16

# 读取图像
fpaths_x, data_x, label_x = read_img(path_x)
# x为数据，y为标签

num_classes = len(set(label_x))

np.random.seed(116)     #原116
np.random.shuffle(data_x)
np.random.seed(116)
np.random.shuffle(label_x)

# 划分训练集、测试集和验证集
x_train_x, x_test_x, y_train_x, y_test_x = train_test_split(data_x, label_x,
                                                            test_size=0.2, random_state=1, stratify=label_x)
x_train_x, x_valid_x, y_train_x, y_valid_x = train_test_split(x_train_x, y_train_x,
                                                              test_size=0.25, random_state=1, stratify=y_train_x)

# 定义对应的实际类别名称
# target_names = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6']

target_names = ['0', '1', '2', '3', '4', '5']
# 数据归一化处理
scaler_x = StandardScaler()
x_train_scaler_x = scaler_x.fit_transform(x_train_x.astype(np.float32).reshape(-1, 1)).reshape(-1, pic_size, pic_size, 3)
x_valid_scaler_x = scaler_x.fit_transform(x_valid_x.astype(np.float32).reshape(-1, 1)).reshape(-1, pic_size, pic_size, 3)
x_test_scaler_x = scaler_x.fit_transform(x_test_x.astype(np.float32).reshape(-1, 1)).reshape(-1, pic_size, pic_size, 3)


# 定义两个输入层
input_layer_x = Input(shape=(pic_size, pic_size, 3))

dropout_x = Dropout(0.5)(input_layer_x)

# 定义卷积神经网络
conv1_x = Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')(input_layer_x)
pool1_x = MaxPooling2D(pool_size=2)(conv1_x)
dropout1_x = Dropout(0.5)(pool1_x)

conv2_x = Conv2D(filters=256, kernel_size=3, padding='same', activation='relu')(dropout1_x)
pool2_x = MaxPooling2D(pool_size=2)(conv2_x)
dropout2_x = Dropout(0.5)(pool2_x)

conv3_x = Conv2D(filters=128, kernel_size=3, padding='same', activation='relu')(dropout2_x)
pool3_x = MaxPooling2D(pool_size=2)(conv3_x)
dropout3_x = Dropout(0.5)(pool3_x)

flat_x = Flatten()(dropout3_x)

# 全连接层
dense = Dense(64, activation='relu')(flat_x)

# 输出层
output_layer = Dense(num_classes, activation='softmax')(dense)

# 定义模型
model = Model(inputs=[input_layer_x], outputs=output_layer)

# 编译模型
model.compile(loss="sparse_categorical_crossentropy", optimizer="Adam", metrics=["accuracy"])

# 输出模型结构
model.summary()

# 模型训练
history = model.fit([x_train_scaler_x], y_train_x, epochs=100, validation_data=([x_valid_scaler_x], y_valid_x))
# TODO 此处 y_train_x y_valid_x 由于是标签 所以 _x和_y都可以

# 画出模型在训练过程中的损失函数和准确率的变化
# plot_learning(history)

plt.plot(history.history['loss'], label='训练集损失函数', marker='o', linestyle='--', markerfacecolor='none')
plt.plot(history.history['val_loss'], label='验证集损失函数', marker='s', linestyle='-.', markerfacecolor='none')
plt.xlabel('训练迭代数')
# plt.ylabel('Loss')

# 绘制训练精度和验证精度
plt.plot(history.history['acc'], label='训练集精度', marker='^', linestyle=':', markerfacecolor='none')
plt.plot(history.history['val_acc'], label='验证集精度', marker='x', linestyle='-', markerfacecolor='none')

# 添加图例
plt.legend()

plt.ylim(0, 1)

# 显示图形
plt.show()


# 模型预测
predict = model.predict([x_test_scaler_x])
predict = np.argmax(predict, axis=1)

# 输出测试集的精确率、召回率和 F1 测度等指标
print(classification_report(y_test_x, predict, target_names=target_names))
# TODO 此处 y_test_x 由于是标签 所以 _x和_y都可以

# 绘制混淆矩阵黑白
conmatrix = confusion_matrix(y_true=y_test_x, y_pred=predict)
plot_confusion_matrix(conmatrix)
print(conmatrix)


# 绘制混淆矩阵彩色
df_cm = pd.DataFrame(conmatrix,
                     index=target_names,
                     columns=target_names)
plt.figure()
# plt.yticks(rotation=90)
# plt.title('混淆矩阵对应正误个数图')
sn.heatmap(df_cm, annot=True, cmap="magma", fmt='g')
plt.show()

# # 绘制混淆矩阵彩色
# df_cm = pd.DataFrame(conmatrix,
#                      index=target_names,
#                      columns=target_names)
# # plt.figure(figsize=(11, 11))
# plt.figure()
# # plt.title('混淆矩阵对应正误个数图')
# sn.heatmap(df_cm, annot=True, cmap="magma", fmt='g')
# plt.show()