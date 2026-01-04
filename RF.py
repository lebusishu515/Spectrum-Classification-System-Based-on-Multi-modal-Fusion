import pandas as pd
import seaborn as sn
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib
from sklearn.metrics import hamming_loss

# 绘制混淆矩阵
def plot_confusion_matrix(cm, title='混淆矩阵热力图', cmap=plt.cm.binary):
    plt.figure(figsize=(11, 11))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title('混淆矩阵热力图', fontdict={'weight': 'normal', 'size': 30})
    plt.colorbar()
    xlocations = np.array(range(6))
    plt.xticks(xlocations, it, rotation=0)
    plt.ylabel('正确标签', font)
    plt.xlabel('预测标签', font)
    plt.savefig('混淆矩阵热力图.jpg')
    plt.show()

#对五类数据，定义字典
def Iris_label(s):
    it = {b'bomo': 0, b'bomf': 1, b'wifif': 2, b'wifib': 3, b'fm': 4, b'noise': 5}
    return it[s]

#字体显示
matplotlib.rcParams['font.family'] = 'SimHei'

#读入数据
path = 'featurepoint_radar_2_Input.txt'
data = np.loadtxt(path, dtype=float, delimiter='	', converters={10:Iris_label})
# converters={22:Iris_label}中“22”指的是第23列：将第23列的str转化为label(number)
# x为数据，y为标签
x, y = np.split(data, indices_or_sections=(10,), axis=1)
np.random.seed(66)

# 划分数据、标签
train_data, test_data, train_label, test_label = train_test_split(x, y, random_state=1, train_size=0.8,test_size=0.2)  

#决策树为150棵，寻找最佳分割时需要考虑的特征数目为Auto
RF = RandomForestClassifier(n_estimators=150, max_features="sqrt")
np.random.seed(66)

RF.fit(train_data, train_label.ravel())
#print(np.shape(train_data),np.shape(train_label.ravel()))

print("RF训练集精度：", RF.score(train_data, train_label))
print("RF测试集精度：", RF.score(test_data, test_label))

test_total, test_total_label = np.split(data, indices_or_sections=(10,), axis=1)  
# x为数据，y为标签
predict_label_RF=RF.predict(test_data)
#对测试集展开预测并保存
print("Test Index：\n", sklearn.metrics.classification_report(test_label, predict_label_RF))
#输出测试集测试得到的主要参数列表：精度、召回率、F1 分度值、记录条数、宏平均、微平均、加权平均。
print('kappa_RF:',sklearn.metrics.cohen_kappa_score(test_label, predict_label_RF))
#衡量分类水平的随机标签

it = [0,1,2,3,4,5]

font = {
'weight' : 'normal',
'size'   : 30,
}




# 绘制混淆矩阵
conmatrix = confusion_matrix(y_true=test_label, y_pred=predict_label_RF)
plot_confusion_matrix(conmatrix)
print(conmatrix)

ham_distance = hamming_loss(test_label,predict_label_RF)
print(ham_distance)

# sns.set()
# f,ax = plt.subplots()
# y_true = test_label
# y_pred = predict_label_RF
# C2 = confusion_matrix(y_true, y_pred, it)
# tick_marks = np.arange(6)
# plt.xticks(tick_marks, it, rotation=0)
# plt.yticks(tick_marks, it)
# plt.imshow(C2, interpolation='nearest')
# sns.heatmap(C2, annot=True, fmt='d',xticklabels='auto', yticklabels='auto', ax=ax)  # 画热力图
# '''
# ax.set_title('混淆矩阵热力图', fontsize=20) #标题
# ax.set_ylabel('正确标签', fontsize=15)
# ax.set_xlabel('预测标签', fontsize=15)
# '''



# # 绘制混淆矩阵彩色
# df_cm = pd.DataFrame(conmatrix,
#                      index=it,
#                      columns=it)
# plt.figure(figsize=(11, 11))
# # plt.yticks(rotation=90)
# plt.title('混淆矩阵对应正误个数图')
# sn.heatmap(df_cm, annot=True, cmap="BuPu")
# plt.show()


df_cm = pd.DataFrame(conmatrix,
                     index=it,
                     columns=it)
# plt.figure(figsize=(6, ))
plt.figure( )
# plt.title('混淆矩阵对应正误个数图')
sn.heatmap(df_cm, annot=True, cmap="magma", fmt='g')
plt.show()