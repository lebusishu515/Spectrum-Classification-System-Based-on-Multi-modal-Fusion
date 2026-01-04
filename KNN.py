import pandas as pd
import seaborn as sn
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sklearn
import seaborn as sns
from sklearn.metrics import hamming_loss
from sklearn.metrics import jaccard_similarity_score
import importlib

# 定义几种类别字典
def Iris_label(s):
    it = {b'bomo': 0, b'bomf': 1, b'wifif': 2, b'wifib': 3, b'fm': 4, b'noise': 5}
    return it[s]

def plot_confusion_matrix(cm, title='混淆矩阵热力图', cmap=plt.cm.binary):
    plt.figure(figsize=(11, 11))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title('混淆矩阵热力图',fontdict={'weight':'normal','size': 30})
    plt.colorbar()
    xlocations = np.array(range(6))
    plt.xticks(xlocations, it, rotation=0)
    plt.ylabel('正确标签', font)
    plt.xlabel('预测标签', font)
    plt.savefig('混淆矩阵热力图.jpg')
    plt.show()

# rcParams是一个字典，其中有各种参数，涉及字体的有以下几种
matplotlib.rcParams['font.family'] = 'SimHei'

target_names = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6']

# 定义路径
path = 'featurepoint_radar_2_Input.txt'
data = np.loadtxt(path, dtype=float, delimiter='	', converters={10: Iris_label})
# converters={22:Iris_label}中“22”指的是第23列：将第23列的str转化为label(number)

# x为数据，y为标签
x, y = np.split(data, indices_or_sections=(10,), axis=1)  
# 划分数据集和验证集

np.random.seed(66)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=32, train_size=0.8, test_size=0.2)
# 设计算法参数
clf = KNeighborsClassifier(n_neighbors=8)
clf.fit(x_train, y_train)
print(np.shape(x_train))
# 调用sklearn库函数，进行分类
pre_test = clf.predict(x_test)
acc = accuracy_score(y_test, pre_test)
# 调用精确率函数
print("Test Index：", sklearn.metrics.classification_report(y_test, pre_test))
#输出测试集测试得到的主要参数列表：精度、召回率、F1 分度值、记录条数、宏平均、微平均、加权平均。
print('准确率', acc)

font = {
'weight': 'normal',
'size': 30,
}

it = [0, 1, 2, 3, 4, 5]



conmatrix = confusion_matrix(y_true=y_test, y_pred=pre_test)
plot_confusion_matrix(conmatrix)
print(conmatrix)
print('准确率',acc)



# sns.set()
# f,ax = plt.subplots()
# y_true = y_test
# y_pred = pre_test
# C2 = confusion_matrix(y_true, y_pred, it)
# tick_marks = np.arange(6)
# plt.xticks(tick_marks, it, rotation=0)
# plt.yticks(tick_marks, it)
# plt.imshow(C2, interpolation='nearest')
# sns.heatmap(C2, annot=True, fmt='d',xticklabels='auto', yticklabels='auto', ax=ax)  # 画热力图
'''
ax.set_title('混淆矩阵热力图', fontsize=20) #标题
ax.set_ylabel('正确标签', fontsize=15)
ax.set_xlabel('预测标签', fontsize=15)
'''

ham_distance = hamming_loss(y_test, pre_test)
print(ham_distance)


# 绘制混淆矩阵彩色
df_cm = pd.DataFrame(conmatrix,
                     index=it,
                     columns=it)
# plt.figure(figsize=(6, ))
plt.figure()
# plt.title('混淆矩阵对应正误个数图')
sn.heatmap(df_cm, annot=True, cmap="magma", fmt='g')
plt.show()
