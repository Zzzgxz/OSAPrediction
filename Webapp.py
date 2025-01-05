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

# 添加描述和分隔线让模型选择更加显眼
st.markdown("""
### Step 1: Choose the Prediction Model
Use the dropdown below to select the model based on your dataset type:
- **PSG Model**: For polysomnography-based data.
- **NPE Model**: For non-polysomnography examination data.
""")
st.divider()  # 分隔线

# 用户选择模型
model_choice = st.selectbox('Choose the model:', ['PSG Model', 'NPE Model'])


# In[2]:


# 根据所选模型显示不同的输入选项
if model_choice == 'PSG Model':
     st.subheader('PSG Model: Feature Input')  # 添加子标题
    data_1 = pd.read_csv('shhssmote.csv')
    selected_features = ['AGE', 'HEIGHT', 'WEIGHT', 'MINSPO2', 'AUGSPO2', 'ODI', 'TG', 'SLEEPEFFICIENCY', 'HDL-C']
    feature_inputs = []
    # 根据特征类型设置不同的输入框
    for feature in selected_features:
        if feature == 'AGE':
            val = st.number_input(f'Enter {feature}', min_value=0, max_value=120, step=1, value=30)  # 整数输入框
        elif feature in ['HEIGHT', 'WEIGHT', 'MINSPO2', 'AUGSPO2', 'TG', 'SLEEPEFFICIENCY', 'HDL-C']:
            val = st.number_input(f'Enter {feature}', min_value=0.0, step=0.1, value=0.0)  # 浮点数输入框
        elif feature == 'ODI':
            val = st.slider(f'Enter {feature}', min_value=0, max_value=100, value=50)  # 滑动条
        else:
            val = st.text_input(f'Enter {feature}', '')  # 默认文本框
        feature_inputs.append(val)
    
    # 模型定义和训练逻辑
    X = data_1[selected_features]
    y = data_1['OSA']
    model = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=3), n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X, y)
    
elif model_choice == 'NPE Model':
    st.subheader('NPE Model: Feature Input')  # 添加子标题
    data_2 = pd.read_csv('ZJU4HSMOTEDATA.csv')
    selected_features = ["wbc", "RBC", "TC", "HDL-C", "UA", "smoke", "HTN", "GENDER", "AGE", "bmi"]
    feature_inputs = []
    # 定义单位
    units = {
        "wbc": "×10⁹/L",
        "RBC": "×10¹²/L",
        "TC": "mmol/L",
        "HDL-C": "mmol/L",
        "UA": "μmol/L",
        "AGE": "years",
        "bmi": "kg/m²"
    }

    # 根据特征类型设置不同的输入框
    for feature in selected_features:
        if feature in ['smoke', 'HTN']:  # 0/1 变量
            val = st.selectbox(f'Enter {feature} (0 for No, 1 for Yes)', options=[0, 1], index=0)
        elif feature == 'GENDER':  # 性别，假设 0 为男性，1 为女性
            val = st.selectbox(f'Enter {feature} (0 for Male, 1 for Female)', options=[0, 1], index=0)
        elif feature == 'AGE':  # 年龄，整数输入
            val = st.number_input(f'Enter {feature} ({units["AGE"]})', min_value=0, max_value=120, step=1, value=30)
        elif feature in ['wbc', 'RBC', 'TC', 'HDL-C', 'UA', 'bmi']:  # 浮点数输入
            val = st.number_input(f'Enter {feature} ({units[feature]})', min_value=0.0, step=0.1, value=0.0)
        else:  # 默认处理
            val = st.text_input(f'Enter {feature}', '')

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




