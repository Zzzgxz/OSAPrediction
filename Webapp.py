#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import streamlit as st
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold

# 假设 data_resampled 已经被加载
# data_resampled = pd.read_csv('your_data.csv')

# 初始化 Streamlit 应用
st.title('OSA Prediction Model')

# 用户选择模型
model_choice = st.selectbox('Choose the model:', ['PSG Model', 'NPE Model'])


# In[2]:


# 根据所选模型显示不同的输入选项
if model_choice == 'PSG Model':
    data_1 = pd.read_csv('shhssmote.csv')
    selected_features = ['AGE', 'HEIGHT', 'WEIGHT', 'MINSPO2', 'AUGSPO2', 'ODI', 'TG', 'SLEEPEFFICIENCY', 'HDL-C']
    feature_inputs = []
    for feature in selected_features:
        val = st.number_input(f'Enter {feature}', value=0.0)
        feature_inputs.append(val)
    
    # 模型定义和训练逻辑
    X = data_1[selected_features]
    y = data_1['OSA']
    model = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=3), n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X, y)
    
elif model_choice == 'NPE Model':
    data_2 = pd.read_csv('ZJU4HSMOTEDATA.csv')
    selected_features = ["wbc", "RBC", "TC", "HDL-C", "UA", "smoke", "HTN", "GENDER", "AGE", "bmi"]
    feature_inputs = []
    for feature in selected_features:
        val = st.number_input(f'Enter {feature}', value=0.0)
        feature_inputs.append(val)

    # 模型定义和训练逻辑
    X = data_2[selected_features]
    y = data_2['OSA']
    model = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=3), n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X, y)


# In[3]:


# 预测按钮
if st.button('Predict'):
    # 输入特征转换为 numpy 数组
    input_features = np.array([feature_inputs])
    # 进行预测
    prediction = model.predict(input_features)
    if prediction[0] == 0:
        st.success('Low risk of OSA')
    else:
        st.error('High risk of OSA')


# In[ ]:




