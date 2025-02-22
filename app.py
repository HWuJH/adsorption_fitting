import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# 语言吸附等温线方程
def langmuir_single(P, qm, b):
    return (qm * b * P) / (1 + b * P)

def langmuir_dual(P, q1, b1, q2, b2):
    return (q1 * b1 * P) / (1 + b1 * P) + (q2 * b2 * P) / (1 + b2 * P)

# 1️⃣ 上传 CSV 文件
st.title("吸附等温线拟合")
uploaded_file = st.file_uploader("上传 CSV 文件", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()  # 确保列名无空格

    # 显示 CSV 数据
    st.write("CSV 数据预览：", df.head())

    # 按钮选择 CO₂ 或 N₂
    option = st.radio("选择气体进行拟合", ["CO₂", "N₂"])

    if option == "CO₂":
        # 2️⃣ 处理 CO₂ 数据（去掉 NaN）
        df_clean = df.dropna(subset=["CO2_ads"])
        P = df_clean["Pressure"].astype(float).values
        q_CO2 = df_clean["CO2_ads"].astype(float).values

        # 3️⃣ 进行拟合
        try:
            popt, _ = curve_fit(langmuir_dual, P, q_CO2, p0=[1, 1, 1, 1])
            q1, b1, q2, b2 = popt
            st.write(f"**拟合参数（CO₂ 双位点吸附）：**")
            st.write(f"q1 = {q1:.4f}, b1 = {b1:.4f}, q2 = {q2:.4f}, b2 = {b2:.4f}")

            # 计算 R²
            q_pred = langmuir_dual(P, *popt)
            R2 = 1 - np.sum((q_CO2 - q_pred) ** 2) / np.sum((q_CO2 - np.mean(q_CO2)) ** 2)
            st.write(f"拟合优度 R² = {R2:.4f}")

            # 4️⃣ 绘制拟合曲线
            plt.figure(figsize=(6, 4))
            plt.scatter(P, q_CO2, label="实验数据", color="red")
            plt.plot(P, q_pred, label="拟合曲线", color="blue")
            plt.xlabel("压力 P")
            plt.ylabel("吸附量 q")
            plt.title("CO₂ 双位点吸附拟合")
            plt.legend()
            st.pyplot(plt)

        except Exception as e:
            st.error(f"拟合失败: {e}")

    elif option == "N₂":
        # 2️⃣ 处理 N₂ 数据（去掉 NaN）
        df_clean = df.dropna(subset=["N2_ads"])
        P = df_clean["Pressure"].astype(float).values
        q_N2 = df_clean["N2_ads"].astype(float).values

        # 3️⃣ 进行拟合
        try:
            popt, _ = curve_fit(langmuir_single, P, q_N2, p0=[1, 1])
            qm, b = popt
            st.write(f"**拟合参数（N₂ 单位点吸附）：**")
            st.write(f"qm = {qm:.4f}, b = {b:.4f}")

            # 计算 R²
            q_pred = langmuir_single(P, *popt)
            R2 = 1 - np.sum((q_N2 - q_pred) ** 2) / np.sum((q_N2 - np.mean(q_N2)) ** 2)
            st.write(f"拟合优度 R² = {R2:.4f}")

            # 4️⃣ 绘制拟合曲线
            plt.figure(figsize=(6, 4))
            plt.scatter(P, q_N2, label="实验数据", color="green")
            plt.plot(P, q_pred, label="拟合曲线", color="orange")
            plt.xlabel("压力 P")
            plt.ylabel("吸附量 q")
            plt.title("N₂ 单位点吸附拟合")
            plt.legend()
            st.pyplot(plt)

        except Exception as e:
            st.error(f"拟合失败: {e}")
