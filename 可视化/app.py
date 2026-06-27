# -*- coding: utf-8 -*-
"""
糖尿病患病风险智能筛查系统 - Flask版
"""

from flask import Flask, render_template, request, jsonify, session
import pandas as pd
import numpy as np
import os
import pickle
import base64
import io
import uuid
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import shap
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.secret_key = os.urandom(24)

# 存储每个会话的预测结果和对话历史
session_predictions = {}
session_chat_history = {}

# ==================== 路径配置 ====================
# 基于脚本位置自动定位项目根目录（可视化/ 的上级目录）
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
CHART_DIR = os.path.join(OUTPUT_DIR, "charts")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")

# ==================== 中文字体 ====================
try:
    font_paths = ['C:/Windows/Fonts/msyh.ttc', 'C:/Windows/Fonts/simhei.ttf']
    for fp in font_paths:
        if os.path.exists(fp):
            matplotlib.rcParams['font.family'] = FontProperties(fname=fp).get_name()
            break
except:
    pass
matplotlib.rcParams['axes.unicode_minus'] = False

# ==================== 加载模型 ====================
print("加载模型...")
model_path = os.path.join(MODEL_DIR, "catboost_final.pkl")
with open(model_path, 'rb') as f:
    model = pickle.load(f)
print(f"✓ 模型加载完成: {model.__class__.__name__}")

# ==================== 加载Qwen大模型 ====================
QWEN_MODEL_DIR = r"D:\linshi\本地小模型"
QWEN_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
QWEN_MODEL_PATH = os.path.join(QWEN_MODEL_DIR, "Qwen2.5-1.5B-Instruct")
qwen_model = None
qwen_tokenizer = None

print("\n" + "=" * 60)
print("Qwen AI大模型 - 健康对话模块")
print("=" * 60)

# 检测模型是否存在
if os.path.exists(os.path.join(QWEN_MODEL_PATH, "config.json")):
    print(f"✓ 模型已存在: {QWEN_MODEL_PATH}")
else:
    print(f"⚠ 未检测到Qwen模型")
    print(f"  模型：{QWEN_MODEL_NAME}")
    print(f"  大小：约3.6GB")
    print(f"  来源：HuggingFace（国际开源模型托管平台，开源可审计）")
    print()
    user_choice = input("是否下载Qwen模型以启用AI健康对话功能？(y/n): ").strip().lower()
    if user_choice == 'y' or user_choice == 'yes':
        print("\n开始下载...")
        try:
            from huggingface_hub import snapshot_download
            os.makedirs(QWEN_MODEL_DIR, exist_ok=True)
            snapshot_download(
                repo_id=QWEN_MODEL_NAME,
                local_dir=QWEN_MODEL_PATH,
            )
            print("✓ 下载完成！")
        except Exception as e:
            print(f"✗ 下载失败: {e}")
            print("  AI对话功能将不可用，预测功能不受影响。")
    else:
        print("\n跳过下载，AI对话功能将不可用，预测功能不受影响。")

print()
print("加载模型到内存...")

try:
    qwen_tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_PATH, trust_remote_code=True)
    qwen_model = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    qwen_model.eval()
    device = next(qwen_model.parameters()).device
    print(f"✓ Qwen大模型加载完成: {QWEN_MODEL_PATH}")
    print(f"  运行设备: {device}")
except Exception as e:
    print(f"⚠ Qwen大模型加载失败: {e}")
    print("  AI对话功能将不可用，风险预测功能不受影响。")
    print("  如需AI对话功能，请确保：")
    print("  1. 已安装 bitsandbytes: pip install bitsandbytes")
    print("  2. Windows页面文件 >= 8GB（系统设置→虚拟内存）")

# 加载实验数据
def load_csv(name):
    path = os.path.join(OUTPUT_DIR, name)
    if os.path.exists(path):
        df = pd.read_csv(path, encoding='utf-8-sig', index_col=0)
        df.index.name = '模型'
        return df.reset_index().to_dict(orient='records')
    return []

# 图片转base64
def img_to_base64(path):
    if not os.path.exists(path):
        return ""
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode()

# ==================== 路由 ====================
@app.route('/')
def index():
    # 加载所有图表
    charts = {}
    chart_files = {
        # 描述性分析
        'eda01': '01_年龄分层患病率.png',
        'eda02': '02_性别患病率对比.png',
        'eda03': '03_BMI分布直方图.png',
        'eda04': '04_高血压高胆固醇患病率.png',
        'eda05': '05_吸烟患病率对比.png',
        'eda06': '06_体育活动患病率.png',
        'eda07': '07_蔬果摄入患病率.png',
        'eda08': '08_BMI高血压交叉分析.png',
        'eda09': '09_高龄吸烟交叉分析.png',
        'eda10': '10_运动肥胖交叉分析.png',
        'eda11': '11_相关性热力图.png',
        'eda12': '12_Top10相关特征.png',
        # 模型对比
        'roc': 'ROC曲线对比.png',
        'pr': 'PR曲线对比.png',
        'metrics': '核心指标横向对比.png',
        'cm_lr': '混淆矩阵_逻辑回归.png',
        'cm_rf': '混淆矩阵_随机森林.png',
        'cm_xgb': '混淆矩阵_XGBoost.png',
        'cm_lgb': '混淆矩阵_LightGBM.png',
        'cm_cb': '混淆矩阵_CatBoost.png',
        'cm_dt': '混淆矩阵_决策树.png',
        'cm_svm': '混淆矩阵_SVM.png',
        # 模型调优
        'shap_bar': 'SHAP_全局特征重要性.png',
        'shap_summary': 'SHAP_特征影响摘要.png',
        'shap_single': 'SHAP_单样本解释.png',
    }
    for key, filename in chart_files.items():
        charts[key] = img_to_base64(os.path.join(CHART_DIR, filename))

    # 加载实验数据
    exp_data = {
        'model_compare': load_csv('测试集评估结果.csv'),
        'imbalance': load_csv('类别不平衡方案对比.csv'),
        'tune_compare': load_csv('调参前后对比.csv'),
        'shap_importance': load_csv('SHAP特征重要性.csv'),
        'pr_ap': load_csv('PR曲线AP值对比.csv'),
        'cv': load_csv('交叉验证结果.csv'),
        'overall_rank': load_csv('模型综合排名.csv'),
        'desc_stats': load_csv('descriptive_stats.csv'),
    }

    # 加载最优参数
    best_params = {}
    params_path = os.path.join(MODEL_DIR, "best_params.pkl")
    if os.path.exists(params_path):
        with open(params_path, 'rb') as f:
            best_params = pickle.load(f)

    return render_template('index.html', charts=charts, exp_data=exp_data, best_params=best_params)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        # 构造输入
        features = [
            'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke',
            'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
            'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
            'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income'
        ]
        # 类别特征需要整数类型
        cat_features = ['HighBP', 'HighChol', 'CholCheck', 'Smoker', 'Stroke',
                        'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
                        'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost',
                        'GenHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income']
        input_values = []
        for f in features:
            val = data.get(f, 0)
            if f in cat_features:
                input_values.append(int(val))
            else:
                input_values.append(float(val))
        input_df = pd.DataFrame([input_values], columns=features)
        # 确保类别特征为int类型
        for f in cat_features:
            input_df[f] = input_df[f].astype(int)

        # 预测
        prob = model.predict_proba(input_df)[:, 1][0]

        # 风险等级
        if prob < 0.3:
            risk = '低风险'
            color = '#009944'
            advice = '🟢 您的患病风险较低，建议保持当前健康的生活方式，每年进行常规体检。'
        elif prob <= 0.7:
            risk = '中风险'
            color = '#ffbb00'
            advice = '🟡 您的患病风险处于中等水平，建议定期监测血糖、血压，加强体育锻炼，控制饮食。'
        else:
            risk = '高风险'
            color = '#dd2222'
            advice = '🔴 您的患病风险较高，建议尽快到内分泌科进行专项检查，遵医嘱进行干预。'

        # SHAP解释
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_df)

        # 生成SHAP图
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[0],
                base_values=explainer.expected_value,
                data=input_df.values[0],
                feature_names=features
            ),
            show=False, max_display=15
        )
        plt.title("SHAP特征贡献解释", fontsize=14, fontweight='bold')
        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        shap_img = base64.b64encode(buf.read()).decode()
        plt.close()

        # Top5特征
        feature_imp = sorted(
            zip(features, shap_values[0]),
            key=lambda x: abs(x[1]), reverse=True
        )
        top5 = [{'name': f, 'value': round(float(v), 4), 'direction': '拉高风险' if v > 0 else '降低风险'}
                for f, v in feature_imp[:5]]

        # 保存预测结果到会话（供聊天使用）
        session_id = request.json.get('session_id', str(uuid.uuid4()))
        session_predictions[session_id] = {
            'input': data,
            'probability': round(float(prob), 4),
            'risk_level': risk,
            'top5_features': top5
        }

        return jsonify({
            'success': True,
            'probability': round(float(prob), 4),
            'risk_level': risk,
            'risk_color': color,
            'advice': advice,
            'shap_image': shap_img,
            'top5_features': top5,
            'session_id': session_id
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': '未上传文件'})

        file = request.files['file']
        df = pd.read_csv(file)

        features = [
            'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke',
            'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
            'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
            'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income'
        ]
        cat_features = ['HighBP', 'HighChol', 'CholCheck', 'Smoker', 'Stroke',
                        'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
                        'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost',
                        'GenHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income']

        missing = [f for f in features if f not in df.columns]
        if missing:
            return jsonify({'success': False, 'error': f'缺少特征列: {", ".join(missing)}'})

        # 确保类别特征为int类型
        for f in cat_features:
            if f in df.columns:
                df[f] = df[f].astype(int)

        probs = model.predict_proba(df[features])[:, 1]
        df['患病概率'] = probs
        df['风险等级'] = ['低风险' if p < 0.3 else '中风险' if p <= 0.7 else '高风险' for p in probs]

        results = df[['患病概率', '风险等级']].to_dict(orient='records')
        csv_data = df.to_csv(index=False)

        return jsonify({
            'success': True,
            'results': results[:50],
            'total': len(results),
            'csv_data': csv_data
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/chat', methods=['POST'])
def chat():
    """AI健康对话接口"""
    import time
    # 检查Qwen模型是否可用
    if qwen_model is None:
        return jsonify({'success': True, 'response': 'AI对话功能暂不可用（大模型未成功加载）。请您参考上方的评估结果和建议，或咨询专业医生。'})
    try:
        data = request.json
        user_message = data.get('message', '')
        session_id = data.get('session_id', '')

        # 获取该会话的预测结果
        pred = session_predictions.get(session_id, None)

        # 构造系统提示词
        system_prompt = """你是健康顾问"糖糖"。根据用户的糖尿病风险评估结果和具体健康指标，给出针对性建议。

回答要求：
1. 必须结合用户的具体情况（有哪些疾病、生活习惯）给出个性化建议
2. 如果用户有高血压，要提醒控制情绪、低盐饮食、规律服药
3. 如果用户BMI高/肥胖，要建议控制体重、增加运动
4. 如果用户吸烟，要建议戒烟
5. 如果用户不运动，要建议逐步增加运动量
6. 如果用户年龄较大，要提醒定期体检
7. 如果风险较高，语气要关切但不要恐吓，给出具体可执行的改善方案
8. 不要提及数据中没有的因素（如家族病史、血糖值等）
9. 不诊断不开药，回答简洁通俗，100字以内"""

        # 如果有预测结果，注入完整上下文
        context = ""
        if pred:
            inp = pred['input']
            conditions = []
            if inp.get('HighBP'): conditions.append("高血压")
            if inp.get('HighChol'): conditions.append("高胆固醇")
            if inp.get('Smoker'): conditions.append("吸烟")
            if inp.get('Stroke'): conditions.append("曾患中风")
            if inp.get('HeartDiseaseorAttack'): conditions.append("心脏病")
            if not inp.get('PhysActivity'): conditions.append("不运动")
            if not inp.get('Fruits'): conditions.append("不吃水果")
            if not inp.get('Veggies'): conditions.append("不吃蔬菜")
            if inp.get('HvyAlcoholConsump'): conditions.append("重度饮酒")
            if inp.get('DiffWalk'): conditions.append("行走困难")
            if inp.get('NoDocbcCost'): conditions.append("因费用看不起病")

            cond_text = "、".join(conditions) if conditions else "无明显异常"
            context = f"""用户档案：
- 风险等级：{pred['risk_level']}，患病概率：{pred['probability']*100:.0f}%
- 年龄：{inp.get('Age','?')}，性别：{'男' if inp.get('Sex') else '女'}
- BMI：{inp.get('BMI','?')}
- 已有状况：{cond_text}
- 自评健康：{inp.get('GenHlth','?')}/5
- 过去30天不健康天数：身体{inp.get('PhysHlth',0)}天，心理{inp.get('MentHlth',0)}天
"""

        # 获取历史对话（只保留最近2轮，避免过长）
        history = session_chat_history.get(session_id, [])
        history = history[-4:]  # 最多保留2轮对话

        # 构造消息
        messages = [
            {"role": "system", "content": system_prompt + context}
        ] + history + [
            {"role": "user", "content": user_message}
        ]

        # 保存对话历史
        history.append({"role": "user", "content": user_message})

        # 调用Qwen模型
        text = qwen_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = qwen_tokenizer([text], return_tensors="pt").to(qwen_model.device)

        with torch.no_grad():
            t1 = time.time()
            outputs = qwen_model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=qwen_tokenizer.eos_token_id,
                eos_token_id=qwen_tokenizer.eos_token_id
            )
            t2 = time.time()
            print(f"[Qwen] 生成耗时: {t2-t1:.1f}s, tokens: {outputs.shape[1]-inputs.input_ids.shape[1]}")

        # 解码输出
        generated_ids = outputs[0][inputs.input_ids.shape[1]:]
        response = qwen_tokenizer.decode(generated_ids, skip_special_tokens=True)

        # 保存助手回复到历史
        history.append({"role": "assistant", "content": response})
        session_chat_history[session_id] = history

        return jsonify({'success': True, 'response': response})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("糖尿病患病风险智能筛查系统")
    print("访问地址: http://localhost:5000")
    print("=" * 60)
    app.run(debug=False, port=5000)
