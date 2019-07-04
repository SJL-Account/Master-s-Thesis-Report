
# coding: utf-8

# In[1]:

#-------------------基础工具------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.signal
import time
import os
#------------------sklearn-------------------------------
from sklearn.cluster import k_means
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.manifold import t_sne
import  sklearn.learning_curve 
from sklearn.tree import DecisionTreeClassifier

#------------------keras --------------------------------
from keras.utils import plot_model
from keras.layers import Dropout
import keras 
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Conv1D
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import MaxPool2D

from keras.optimizers import SGD
from keras.optimizers import Adam
from keras import Sequential


#-----------------xgboost----------------------------

import xgboost as xgb

#---------------------参数设置----------------------------
get_ipython().magic('matplotlib inline')
#CPU还是GPU
import copy
namespace='原始特征抽样_卷积'
name_path='原始特征抽样_卷积/'


# In[2]:

def calc_kappa(y_true,y_pre):
    p0=0
    pe=0
    confux_m=confusion_matrix(y_true,y_pre)
    confux_m_sum=confux_m.sum()
    for i in range(len(confux_m)):
        p0+=confux_m[i][i]
        pe+=((confux_m[:,i]).sum())*((confux_m[i,:]).sum())
    p0=p0/confux_m_sum
    pe=pe/(confux_m_sum*confux_m_sum)
    kappa=(p0-pe)/(1-pe)
    return kappa


# In[3]:

def conv_custom_1(data,columns,kernel,op='arg'):
    columns=data.columns[2:-1].tolist()
    label_img=data['type'].values.reshape(256,108)
    for col in columns:
        var_img=data[col].values.reshape(256,108)
        con_var_img=np.zeros((258,110))
        con_label_img=np.ndarray(shape=(258,110),dtype=np.int32)
        con_label_img[:,:]=-1
        con_re=np.zeros((256,108))
        for i in range(256):
            for j in range(108):
                con_var_img[i+1][j+1]=var_img[i][j]
                con_label_img[i+1][j+1]=label_img[i][j]
        for i in range(1,257):
            for j in range(1,109):
                center=con_label_img[i][j]
                label_map=con_label_img[i-1:i+2,j-1:j+2]
                label_filter=(label_map==center)
                
                con_re[i-1][j-1]=(((con_var_img[i-1:i+2,j-1:j+2][label_filter])*(kernel[label_filter])).sum())
                                 
        if op=='arg':
            data[col+'_arg']=con_re.flatten()
        elif op=='var':
            data[col+'_var']=con_re.flatten()


# In[4]:


def conv_(data,columns,kernel):
    for col in columns:
        var_img=data[col].values.reshape(256,108)
        data[col+'_arg']=scipy.signal.convolve2d(var_img,kernel,mode='same').flatten()


# In[5]:

def output(pipline_fit_score,pipline_test_score,pipline_fit_test_score,kappas_,y_true,y_pre,y_fit_pre,clf_name):


    if  not (os.path.exists(name_path+clf_name)):

        os.mkdir(name_path+clf_name)


    pipline_fit_score['accuracy'].append(accuracy_score(y_true,y_fit_pre))

    pipline_test_score['accuracy'].append(accuracy_score(y_true,y_pre))

    kappas_['kappa'].append(calc_kappa(y_true,y_pre))

    pipline_fit_test_score['fit-accuracy'].append(accuracy_score(y_true,y_fit_pre))

    pipline_fit_test_score['test-accuracy'].append(accuracy_score(y_true,y_pre))

    pd.DataFrame(confusion_matrix(y_true,y_pre),index=range(1,17),columns=range(1,17)).to_csv(name_path+clf_name+'/'+clf_name+'_confux_matrix.csv')  

    Salinas_df5_raw[y_true!=y_pre][['row','col']].to_csv(name_path+clf_name+'/'+clf_name+'false_loc.csv',index=None,columns=None)

    Salinas_df5_raw['pre_type']=y_pre
    
    #np.savetxt(clf_name+'_pre_label.grd', y_pre.reshape(256,108),fmt='%d')

    Salinas_df5_raw[['row','col','pre_type']].to_csv( name_path+clf_name+'/'+clf_name+'_pre.csv')


# In[6]:

def ouput2(pipline_fit_score,pipline_test_score,pipline_fit_test_score,kappas_,clf_names):
    
    if not (os.path.exists(name_path+'统计量')):
        
        os.mkdir(name_path+'统计量')
    
    print('正在输出分类器准确率')

    pd.DataFrame(pipline_fit_score,index=clf_names).sort_values(by='accuracy').to_csv(name_path+'统计量/'+clf_name+'fit_accuracy.csv')
    pd.DataFrame(pipline_test_score,index=clf_names).sort_values(by='accuracy').to_csv(name_path+'统计量/'+clf_name+'test_accuracy.csv')
    pd.DataFrame(pipline_fit_test_score,index=clf_names).to_csv(name_path+'统计量/'+clf_name+'fit_test_accuracy.csv')

    print('正在输出kappa系数')

    pd.DataFrame(kappas_,index=clf_names).sort_values(by='kappa').to_csv(name_path+'统计量/'+clf_name+'kappa.csv')


# 数据的导入，切分，标准化

# In[7]:

Salinas_df4_raw=pd.read_csv('Salinas_bandselect/SalinasResize_FCwithRC_No_4.txt',delimiter='\t',index_col=False)
Salinas_df5_raw=pd.read_csv('Salinas_bandselect/SalinasResize_FCwithRC_No_5.txt',delimiter='\t',index_col=False)
df4_columns=Salinas_df4_raw.columns[2:-1].tolist()
df5_columns=Salinas_df5_raw.columns[2:-1].tolist()
df4_arg_columns=[]
df5_arg_columns=[]
df4_var_columns=[]
df5_var_columns=[]
df4_var_var_columns=[]
df5_var_var_columns=[]
for col in df4_columns:
    df4_arg_columns.append(col+'_arg')
    Salinas_df4_raw[col+'_arg']=0
for col in df5_columns:
    df5_arg_columns.append(col+'_arg')
    Salinas_df5_raw[col+'_arg']=0

#---------------------------------------------------------
dirs_=os.listdir('../变差函数feature_map/变差函数_25波段/')

for dir_ in dirs_:
    print('正在计算..'+dir_)
    deviation=np.loadtxt('../变差函数feature_map/变差函数_25波段/'+dir_)

    deviation_=np.zeros((42,42))
    deviation_[:-1,:-1]=deviation
    deviation_sum=deviation_.sum()
    conv_matrix=np.zeros((3,3))

    for i in range(3):
        for j in range(3):
            conv_matrix[i,j]=((deviation_[i*14:(i+1)*14,j*14:(j+1)*14]).sum())/deviation_sum

    kernel=conv_matrix
    print(kernel)
    Type=int(dir_[-9])
    
    #创建列

    #复制数据
    Salinas_df4_temp=copy.deepcopy(Salinas_df4_raw)
    Salinas_df5_temp=copy.deepcopy(Salinas_df5_raw)
    
    #做成除了该类型外其他类型的光谱强度均为0的情况，Featuremap
    Salinas_df4_temp[Salinas_df4_temp['type']!=Type][df4_columns]=0
    Salinas_df5_temp[Salinas_df5_temp['type']!=Type][df5_columns]=0
    #获取对应的index
    indexs_4=Salinas_df4_temp[Salinas_df4_temp['type']==Type].index
    indexs_5=Salinas_df5_temp[Salinas_df5_temp['type']==Type].index
    
    #卷积操作
    conv_(Salinas_df4_temp,df4_columns,kernel)
    conv_(Salinas_df5_temp,df5_columns,kernel)
   
    
    #对应类型对应列进行复制
    Salinas_df4_raw.ix[indexs_4,df4_arg_columns]=Salinas_df4_temp.ix[indexs_4,df4_arg_columns]
    Salinas_df5_raw.ix[indexs_5,df5_arg_columns]=Salinas_df5_temp.ix[indexs_5,df5_arg_columns]
#----------------------------------------------------------



Salinas_df4_raw=Salinas_df4_raw[Salinas_df4_raw.type!=0]
Salinas_df5_raw=Salinas_df5_raw[Salinas_df5_raw.type!=0]
Salinas_df4=copy.deepcopy(Salinas_df4_raw)
Salinas_df5=copy.deepcopy(Salinas_df5_raw)
std=StandardScaler()
Salinas_df4.pop('row')
Salinas_df4.pop('col')
y_train=Salinas_df4.pop('type')
x_train=Salinas_df4
x_train=std.fit_transform(x_train)
Salinas_df5.pop('row')
Salinas_df5.pop('col')
y_test=Salinas_df5.pop('type')
x_test=Salinas_df5
x_test=std.fit_transform(x_test)
y_true=y_test
y_train.value_counts().plot(kind='bar')


# In[11]:

if not (os.path.exists(namespace)):
    os.mkdir(namespace)

piplines=[LogisticRegression(),GradientBoostingClassifier(),RandomForestClassifier(),DecisionTreeClassifier(),KNeighborsClassifier(),SVC()]
pipline_fit_score={'accuracy':[]}
pipline_test_score={'accuracy':[]}
pipline_fit_test_score={'fit-accuracy':[],'test-accuracy':[]}
kappas_={'kappa':[]}

clf_names=['逻辑回归','梯度提升决策树','随机森林','决策树','K-临近','SVM']

for clf_name,clf in zip(clf_names,piplines):
    
    
    clf.fit(x_train,y_train)
    
    print('拟合',clf_name,clf.score(x_train,y_train))
    
    print('测试',clf_name,clf.score(x_test,y_test))
    
    y_fit_pre=clf.predict(x_train)

    y_pre=clf.predict(x_test)
    
    output(pipline_fit_score,pipline_test_score,pipline_fit_test_score,kappas_,y_true,y_pre,y_fit_pre,clf_name)

ouput2(pipline_fit_score,pipline_test_score,pipline_fit_test_score,kappas_,clf_names)


# In[12]:

xgb_data_train=xgb.DMatrix(x_train,label=y_train)
xgb_data_test=xgb.DMatrix(x_test,label=y_test)
params={'max_depth':10,'eta':0.1,'silent':0,'objective':'multi:softmax','num_class':17,'alpha':0.5}
bst=xgb.train(params, 
          xgb_data_train, 
          num_boost_round=400, 
          evals=[(xgb_data_train,'train'),(xgb_data_test,'test')], 
          obj=None, 
          feval=None, 
          maximize=False, 
          early_stopping_rounds=None,
          evals_result=None,
          verbose_eval=100,
          xgb_model=None,
          callbacks=None)
xgb_data_test=xgb.DMatrix(x_test)
y_pre=bst.predict(xgb_data_test)
y_true=y_test

y_pre=bst.predict(xgb_data_test)
y_fit_pre=bst.predict(xgb_data_train)
y_true=y_test
accuracy_score(y_true,y_pre)


# In[13]:

clf_name='XGBoost'
clf_names.append(clf_name)
output(pipline_fit_score,pipline_test_score,pipline_fit_test_score,kappas_,y_true,y_pre,y_fit_pre,clf_name)
ouput2(pipline_fit_score,pipline_test_score,pipline_fit_test_score,kappas_,clf_names)


# In[14]:

i=6
input_dim=x_train.shape[1]
output_dim=len(set(y_train.tolist()))
batch_size=16
lr=0.1
epochs=50
hidden_num = input_dim*i
hidden_num_2 = output_dim*i
ohe = OneHotEncoder()
y_train=ohe.fit_transform(np.matrix(y_train.values).T).toarray()
model=Sequential()
model.add(Dense(input_dim=input_dim,units=hidden_num))
model.add(Activation('relu'))
model.add(Dense(input_dim=hidden_num_2,units=output_dim))
optimizer=SGD(lr=lr)
model.compile(loss='mean_squared_error',optimizer=optimizer)
model.fit(x_train,y_train,epochs=epochs,batch_size=batch_size,verbose=1)

y_pre=model.predict_classes(x_test, batch_size=batch_size)
y_fit_pre=model.predict_classes(x_train, batch_size=batch_size)

y_true=y_test
y_pre=y_pre+1
y_fit_pre=y_fit_pre+1


# In[ ]:

clf_name='NN'
clf_names.append(clf_name)
output(pipline_fit_score,pipline_test_score,pipline_fit_test_score,kappas_,y_true,y_pre,y_fit_pre,clf_name)
ouput2(pipline_fit_score,pipline_test_score,pipline_fit_test_score,kappas_,clf_names)


# # 输出部分

# 交叉矩阵(准确率、召回率,F1)

# kappa系数
# 准确率
# kappa计算结果为-1~1，但通常kappa是落在 0~1 间，可分为五组来表示不同级别的一致性：
# 0.0~0.20极低的一致性(slight)、
# 0.21~0.40一般的一致性(fair)、
# 0.41~0.60 中等的一致性(moderate)、
# 0.61~0.80 高度的一致性(substantial)
# 0.81~1几乎完全一致(almost perfect)。
# 

# losss下降过程(运算时间)

# 错的点的坐标

# 预测值的输出

# 神经网络层数图片

# 学习曲线

# In[ ]:



