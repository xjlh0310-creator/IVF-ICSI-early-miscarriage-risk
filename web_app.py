import streamlit as st
import pandas as pd
import pickle
import xgboost
import os

# ==========================================
# 1. 页面基础配置
# ==========================================
st.set_page_config(
    page_title="IVF 妊娠结局预测系统",
    page_icon="👶",
    layout="wide"
)

# 隐藏 Streamlit 默认菜单（可选）
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

st.title("👶 IVF 早期妊娠结局 AI 预测系统")
st.markdown("### 基于 XGBoost 机器学习模型 (5 变量临床版)")
st.info("本系统部署于云端，仅供科研与临床辅助参考。")
st.markdown("---")

# ==========================================
# 2. 加载模型 (核心步骤)
# ==========================================
@st.cache_resource
def load_model():
    # 优先查找当前目录下的模型文件
    # 注意：请确保 xgb_model.pkl 已上传到 GitHub 仓库的根目录
    model_filename = 'xgb_model.pkl'
    
    # 为了兼容可能的子文件夹结构，增加一个检查
    if not os.path.exists(model_filename):
        # 尝试查找上传代码中提到的子文件夹路径（如果用户没有把模型移出来）
        alt_path = "2.训练集构建模型/xgb_model.pkl"
        if os.path.exists(alt_path):
            model_filename = alt_path
        else:
            st.error(f"❌ 严重错误：未找到模型文件。请确保 `xgb_model.pkl` 已上传到 GitHub 仓库根目录！")
            st.stop()
        
    try:
        with open(model_filename, 'rb') as file:
            loaded_model = pickle.load(file)
        return loaded_model
    except Exception as e:
        st.error(f"模型加载失败: {e}")
        st.stop()

model = load_model()

# ==========================================
# 3. 侧边栏：输入患者数据
# ==========================================
with st.sidebar:
    st.header("📝 患者临床指标录入")
    st.markdown("请准确填写以下 5 项指标：")
    
    with st.form("input_form"):
        # 1. Female_age
        female_age = st.number_input("1. 女方年龄 (Female Age)", min_value=20.0, max_value=55.0, value=32.0, step=1.0)
        
        # 2. BMI
        bmi = st.number_input("2. 体重指数 (BMI)", min_value=10.0, max_value=50.0, value=22.5, step=0.1)
        
        # 3. PLT
        plt_val = st.number_input("3. 血小板计数 (PLT)", min_value=10.0, max_value=600.0, value=250.0, step=1.0)
        
        # 4. FSH
        fsh = st.number_input("4. 促卵泡生成素 (FSH)", min_value=0.0, max_value=100.0, value=7.5, step=0.1)
        
        # 5. TSH
        tsh = st.number_input("5. 促甲状腺激素 (TSH)", min_value=0.0, max_value=50.0, value=2.0, step=0.01)
        
        st.markdown("---")
        submitted = st.form_submit_button("🚀 开始预测 (Run Prediction)")

# ==========================================
# 4. 主界面：预测逻辑与结果展示
# ==========================================
if submitted:
    # 构造数据 DataFrame (列名必须与训练时严格一致)
    input_data = {
        'Female_age': female_age,
        'BMI': bmi,
        'PLT': plt_val,
        'FSH': fsh,
        'TSH': tsh
    }
    df_input = pd.DataFrame([input_data])

    # 显示输入数据
    st.subheader("1. 患者数据概览")
    st.dataframe(df_input, use_container_width=True)

    try:
        # 进行预测
        # predict_proba 返回 [[失败概率, 成功概率]]
        prediction_probs = model.predict_proba(df_input)[0]
        success_prob = prediction_probs[1] # 获取第1类（成功/活产）的概率
        
        st.subheader("2. AI 预测分析")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.write("妊娠成功率预估：")
            st.progress(success_prob)
            
            if success_prob > 0.6:
                st.success(f"🎉 预测成功率: **{success_prob*100:.2f}%**")
                st.markdown("✅ **临床提示**：模型评估该患者预后**良好**，属于高成功率群体。")
                st.balloons()
            elif success_prob < 0.4:
                st.error(f"📉 预测成功率: **{success_prob*100:.2f}%**")
                st.markdown("⚠️ **临床提示**：模型评估风险**较高**，建议仔细排查潜在干扰因素。")
            else:
                st.warning(f"⚖️ 预测成功率: **{success_prob*100:.2f}%**")
                st.markdown("🔹 **临床提示**：模型评估为**中等**水平，建议结合医生经验综合判断。")
        
        with col2:
            st.metric(label="活产概率", value=f"{success_prob:.2%}")

    except Exception as e:
        st.error(f"预测发生错误: {str(e)}")
        st.write("可能原因：输入数据格式异常或模型版本不匹配。")

else:
    st.info("👈 请在左侧侧边栏输入数据，并点击“开始预测”按钮。")

st.markdown("---")
st.caption("⚠️ 免责声明：本工具基于 XGBoost 算法构建，仅供科研参考，不可替代医生临床诊断。")