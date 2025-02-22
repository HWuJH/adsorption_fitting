import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# 📌 1. Langmuir 单位点模型
def langmuir_single(P, qm, b):
    return (qm * b * P) / (1 + b * P)

# 📌 2. Langmuir 双位点模型
def langmuir_dual(P, qm1, b1, qm2, b2):
    return (qm1 * b1 * P) / (1 + b1 * P) + (qm2 * b2 * P) / (1 + b2 * P)

# 📌 3. 计算 R²
def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

# 📌 4. Streamlit 页面
st.title("🧪 吸附等温线拟合")
st.write("📊 支持 CO₂（双位点）和 N₂（单位点）Langmuir 拟合")

# 📌 5. 上传 CSV 数据
uploaded_file = st.file_uploader("📂 上传 CSV 文件（格式: Pressure, CO2, N2）", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head())

    # 读取数据
    P = df["Pressure"].values
    q_CO2 = df["CO2"].values
    q_N2 = df["N2"].values

    # 📌 6. 拟合 CO₂ 双位点 Langmuir
    popt_CO2, _ = curve_fit(langmuir_dual, P, q_CO2, p0=[1, 1, 1, 1])
    qm1, b1, qm2, b2 = popt_CO2
    r2_CO2 = r_squared(q_CO2, langmuir_dual(P, *popt_CO2))

    # 📌 7. 拟合 N₂ 单位点 Langmuir
    popt_N2, _ = curve_fit(langmuir_single, P, q_N2, p0=[1, 1])
    qm_N2, b_N2 = popt_N2
    r2_N2 = r_squared(q_N2, langmuir_single(P, *popt_N2))

    # 📌 8. 显示拟合参数
    st.subheader("📈 拟合结果")
    st.write(f"**CO₂ 双位点 Langmuir 模型**")
    st.write(f"- qm1 = {qm1:.4f}, b1 = {b1:.4f}")
    st.write(f"- qm2 = {qm2:.4f}, b2 = {b2:.4f}")
    st.write(f"**R² = {r2_CO2:.4f}**")

    st.write(f"**N₂ 单位点 Langmuir 模型**")
    st.write(f"- qm = {qm_N2:.4f}, b = {b_N2:.4f}")
    st.write(f"**R² = {r2_N2:.4f}**")

    # 📌 9. 绘制图像
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # CO2 拟合曲线
    P_fit = np.linspace(min(P), max(P), 100)
    q_CO2_fit = langmuir_dual(P_fit, *popt_CO2)
    ax[0].scatter(P, q_CO2, color='red', label="实验数据")
    ax[0].plot(P_fit, q_CO2_fit, color='blue', linestyle="--", label="拟合曲线")
    ax[0].set_xlabel("压力 (P)")
    ax[0].set_ylabel("吸附量 (q)")
    ax[0].set_title("CO₂ Langmuir 拟合")
    ax[0].legend()

    # N2 拟合曲线
    q_N2_fit = langmuir_single(P_fit, *popt_N2)
    ax[1].scatter(P, q_N2, color='green', label="实验数据")
    ax[1].plot(P_fit, q_N2_fit, color='blue', linestyle="--", label="拟合曲线")
    ax[1].set_xlabel("压力 (P)")
    ax[1].set_ylabel("吸附量 (q)")
    ax[1].set_title("N₂ Langmuir 拟合")
    ax[1].legend()

    st.pyplot(fig)
