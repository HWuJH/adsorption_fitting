import streamlit as st
import numpy as np
import pandas as pd
import csv
from lmfit import Model
import os
import matplotlib.pyplot as plt

# 定义压力单位转换函数
def convert_pressure(pressure):
    return pressure / 100000  # 将压力从Pa转换为bar

# 双位点Langmuir模型函数
def dual_langmuir_model(P, qmax1, K1, qmax2, K2):
    return (qmax1 * K1 * P) / (1 + K1 * P) + (qmax2 * K2 * P) / (1 + K2 * P)

# 单位点Langmuir模型函数
def langmuir_model(P, qmax, K):
    return (qmax * K * P) / (1 + K * P)

# Streamlit页面布局
st.title('CO2 和 N2 数据拟合与可视化')

# 使用st.columns()将页面分为两列
col1, col2 = st.columns(2)

# 在左侧列上传CO2数据
with col1:
    st.subheader('上传CO2数据')
    uploaded_file_co2 = st.file_uploader("上传CO2数据 CSV 文件", type=["csv"])

# 在右侧列上传N2数据
with col2:
    st.subheader('上传N2数据')
    uploaded_file_n2 = st.file_uploader("上传N2数据 CSV 文件", type=["csv"])

# 展示吸附等温线公式
def plot_formula():
    fig, ax = plt.subplots(figsize=(8, 2))
    
    # 显示第一个公式，适当向上偏移
    ax.text(0.5, 0.7, r"$q = \frac{q_{max} K P}{1 + K P}$", fontsize=20, ha='center', va='center')
    
    # 显示第二个公式，适当向下偏移
    ax.text(0.5, 0.3, r"$q = \frac{q_{max1} K1 P}{1 + K1 P} + \frac{q_{max2} K2 P}{1 + K2 P}$", fontsize=20, ha='center', va='center')

    ax.axis('off')  # 关闭坐标轴
    st.pyplot(fig)

plot_formula()

# 拟合CO2数据
if uploaded_file_co2 is not None:
    data_co2 = pd.read_csv(uploaded_file_co2, header=None)
    data_co2.columns = ['P_CO2', 'q_CO2']  # 假设CO2数据是两列

    st.write("CO2数据：", data_co2)

    P_CO2 = data_co2['P_CO2'].values
    q_CO2 = data_co2['q_CO2'].values
    P_CO2 = convert_pressure(P_CO2)  # 转换为bar

    # 创建模型对象
    model_co2 = Model(dual_langmuir_model)
    model_co2.set_param_hint('qmax1', value=0.01, min=0)
    model_co2.set_param_hint('K1', value=0.01, min=0)
    model_co2.set_param_hint('qmax2', value=0.01, min=0)
    model_co2.set_param_hint('K2', value=0.01, min=0)

    # 拟合数据
    result_co2 = model_co2.fit(q_CO2, P=P_CO2*100000)  # 转换压力为Pa

    # 拟合参数
    qmax1, K1, qmax2, K2 = result_co2.params['qmax1'].value, result_co2.params['K1'].value, result_co2.params['qmax2'].value, result_co2.params['K2'].value
    r_squared_co2 = result_co2.rsquared

    # 显示拟合结果
    st.write(f"CO2拟合结果： qmax1={qmax1:.4f}, K1={K1:.4f}, qmax2={qmax2:.4f}, K2={K2:.4f}, R²={r_squared_co2:.4f}")

    # 绘制拟合曲线
    plt.figure(figsize=(6, 4))
    plt.scatter(P_CO2, q_CO2, color='blue', label='Data (CO2)')
    plt.plot(P_CO2, result_co2.best_fit, color='red', label='Fitted curve')
    plt.xlabel('Pressure (bar)')
    plt.ylabel('Excess Adsorption (mol/kg)')
    plt.legend()
    plt.title('CO2 Excess Adsorption Fitting')
    st.pyplot(plt)

# 拟合N2数据
if uploaded_file_n2 is not None:
    data_n2 = pd.read_csv(uploaded_file_n2, header=None)
    data_n2.columns = ['P_N2', 'q_N2']  # 假设N2数据是两列

    st.write("N2数据：", data_n2)

    P_N2 = data_n2['P_N2'].values
    q_N2 = data_n2['q_N2'].values

    # 创建模型对象
    model_n2 = Model(langmuir_model)
    model_n2.set_param_hint('qmax', value=0.01, min=0)
    model_n2.set_param_hint('K', value=0.01, min=0)

    # 拟合数据
    result_n2 = model_n2.fit(q_N2, P=P_N2)

    # 拟合参数
    qmax_n2, K_n2 = result_n2.params['qmax'].value, result_n2.params['K'].value
    r_squared_n2 = result_n2.rsquared

    # 显示拟合结果
    st.write(f"N2拟合结果： qmax={qmax_n2:.4f}, K={K_n2:.4f}, R²={r_squared_n2:.4f}")

    # 绘制拟合曲线
    plt.figure(figsize=(6, 4))
    plt.scatter(P_N2, q_N2, color='green', label='Data (N2)')
    plt.plot(P_N2, result_n2.best_fit, color='orange', label='Fitted curve')
    plt.xlabel('Pressure (Pa)')
    plt.ylabel('Excess Adsorption (mol/kg)')
    plt.legend()
    plt.title('N2 Excess Adsorption Fitting')
    st.pyplot(plt)

# 导出拟合结果
export_button = st.button('导出拟合结果')
if export_button:
    output_folder = 'D:/Users/ASUS/Desktop/678RASPA2DSL/Fit_Results'
    os.makedirs(output_folder, exist_ok=True)

    # 保存CO2拟合结果
    co2_fit_data = [
        ['qmax1', 'K1', 'qmax2', 'K2', 'R_squared'],
        [qmax1, K1, qmax2, K2, r_squared_co2]
    ]
    co2_output_path = os.path.join(output_folder, 'CO2_fit_results.csv')
    with open(co2_output_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(co2_fit_data)

    # 保存N2拟合结果
    n2_fit_data = [
        ['qmax', 'K', 'R_squared'],
        [qmax_n2, K_n2, r_squared_n2]
    ]
    n2_output_path = os.path.join(output_folder, 'N2_fit_results.csv')
    with open(n2_output_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(n2_fit_data)

    st.write(f"拟合结果已保存到 {output_folder}")
