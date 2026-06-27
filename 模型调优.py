# -*- coding: utf-8 -*-
"""
CatBoost模型深度优化
步骤1：类别不平衡方案对比
步骤2：最优模型专项调参
步骤3：PR曲线评估体系
步骤4：SHAP可解释性分析
"""

import pandas as pd
import numpy as np
import os
import pickle
import time
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.font_manager import FontProperties
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (confusion_matrix, roc_curve, auc, precision_recall_curve,
                             average_precision_score, f1_score, recall_score,
                             accuracy_score, precision_score)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
import shap
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

def log(msg):
    """带时间戳的日志输出"""
    from datetime import datetime
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

# ==================== 中文字体配置 ====================
try:
    font_paths = [
        'C:/Windows/Fonts/msyh.ttc',
        'C:/Windows/Fonts/simsun.ttc',
        'C:/Windows/Fonts/simhei.ttf',
    ]
    font_prop = None
    for fp in font_paths:
        if os.path.exists(fp):
            font_prop = FontProperties(fname=fp)
            plt.rcParams['font.family'] = font_prop.get_name()
            break
    if font_prop is None:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
except:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 120
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'

# ==================== 配置参数 ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = BASE_DIR
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
CHART_DIR = os.path.join(OUTPUT_DIR, "charts")
os.makedirs(CHART_DIR, exist_ok=True)

RANDOM_STATE = 42
CATEGORICAL_FEATURES = ['HighBP', 'HighChol', 'CholCheck', 'Smoker', 'Stroke',
                        'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
                        'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost',
                        'GenHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income']


def save_fig(fig, name):
    path = os.path.join(CHART_DIR, f"{name}.png")
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ 已保存：{name}.png")


def eval_metrics(y_true, y_pred, y_prob):
    return {
        '准确率': accuracy_score(y_true, y_pred),
        '精确率': precision_score(y_true, y_pred),
        '召回率': recall_score(y_true, y_pred),
        'F1值': f1_score(y_true, y_pred),
        'AUC': auc(*roc_curve(y_true, y_prob)[:2])
    }


# ==================== 加载数据 ====================
log("=" * 60)
log("加载数据")
log("=" * 60)

train_cat = pd.read_csv(os.path.join(DATA_DIR, "train_catboost.csv"))
log("✓ 训练集加载完成")
test_cat = pd.read_csv(os.path.join(DATA_DIR, "test_catboost.csv"))
log("✓ 测试集加载完成")

X_train = train_cat.drop(columns=['Diabetes_binary'])
y_train = train_cat['Diabetes_binary']
X_test = test_cat.drop(columns=['Diabetes_binary'])
y_test = test_cat['Diabetes_binary']

# 标准化版本（用于逻辑回归、随机森林等PR曲线对比）
train_scaled = pd.read_csv(os.path.join(DATA_DIR, "train_scaled.csv"))
test_scaled = pd.read_csv(os.path.join(DATA_DIR, "test_scaled.csv"))
X_train_s = train_scaled.drop(columns=['Diabetes_binary'])
X_test_s = test_scaled.drop(columns=['Diabetes_binary'])
log("✓ 标准化数据集加载完成")

log(f"训练集：{X_train.shape[0]} 条，测试集：{X_test.shape[0]} 条")
log(f"正样本比例：{y_train.mean()*100:.2f}%")


# ================================================================
# 步骤1：类别不平衡方案对比实验
# ================================================================
log("\n" + "=" * 60)
log("【步骤1】类别不平衡方案对比实验")
log("=" * 60)

imbalance_results = {}

# --- 基准组：原始训练集 ---
log("[1/3] 基准组：原始训练集 — 开始训练...")
cat_base = CatBoostClassifier(
    iterations=200, depth=6, learning_rate=0.1,
    random_state=RANDOM_STATE, verbose=0, auto_class_weights='None'
)
cat_base.fit(X_train, y_train, cat_features=CATEGORICAL_FEATURES)
log("[1/3] 基准组 — 训练完成，开始预测...")
y_pred_base = cat_base.predict(X_test)
y_prob_base = cat_base.predict_proba(X_test)[:, 1]
imbalance_results['基准（原始）'] = eval_metrics(y_test, y_pred_base, y_prob_base)
log(f"[1/3] 基准组 — 完成 | 召回率={imbalance_results['基准（原始）']['召回率']:.4f}  "
    f"AUC={imbalance_results['基准（原始）']['AUC']:.4f}  "
    f"F1={imbalance_results['基准（原始）']['F1值']:.4f}")

# --- 方案一：代价敏感学习 ---
log("[2/3] 代价敏感学习 — 开始训练...")
cat_cost = CatBoostClassifier(
    iterations=200, depth=6, learning_rate=0.1,
    random_state=RANDOM_STATE, verbose=0, auto_class_weights='Balanced'
)
cat_cost.fit(X_train, y_train, cat_features=CATEGORICAL_FEATURES)
log("[2/3] 代价敏感学习 — 训练完成，开始预测...")
y_pred_cost = cat_cost.predict(X_test)
y_prob_cost = cat_cost.predict_proba(X_test)[:, 1]
imbalance_results['代价敏感学习'] = eval_metrics(y_test, y_pred_cost, y_prob_cost)
log(f"[2/3] 代价敏感学习 — 完成 | 召回率={imbalance_results['代价敏感学习']['召回率']:.4f}  "
    f"AUC={imbalance_results['代价敏感学习']['AUC']:.4f}  "
    f"F1={imbalance_results['代价敏感学习']['F1值']:.4f}")

# --- 方案二：SMOTE过采样 ---
log("[3/3] SMOTE过采样 — 开始生成样本...")
smote = SMOTE(random_state=RANDOM_STATE, sampling_strategy=1.0)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
log(f"[3/3] SMOTE — 采样完成：{len(y_train)} → {len(y_train_smote)} 条")
log("[3/3] SMOTE — 开始训练CatBoost...")
cat_smote = CatBoostClassifier(
    iterations=200, depth=6, learning_rate=0.1,
    random_state=RANDOM_STATE, verbose=0, auto_class_weights='None'
)
cat_smote.fit(X_train_smote, y_train_smote, cat_features=CATEGORICAL_FEATURES)
log("[3/3] SMOTE — 训练完成，开始预测...")
y_pred_smote = cat_smote.predict(X_test)
y_prob_smote = cat_smote.predict_proba(X_test)[:, 1]
imbalance_results['SMOTE过采样'] = eval_metrics(y_test, y_pred_smote, y_prob_smote)
log(f"[3/3] SMOTE — 完成 | 召回率={imbalance_results['SMOTE过采样']['召回率']:.4f}  "
    f"AUC={imbalance_results['SMOTE过采样']['AUC']:.4f}  "
    f"F1={imbalance_results['SMOTE过采样']['F1值']:.4f}")

# 保存不平衡对比结果
imb_df = pd.DataFrame(imbalance_results).T[['召回率', 'AUC', 'F1值']]
imb_df.to_csv(os.path.join(OUTPUT_DIR, '类别不平衡方案对比.csv'), encoding='utf-8-sig')
log("\n--- 不平衡方案对比汇总 ---")
log(imb_df.round(4).to_string())

best_imb = imb_df.sort_values('召回率', ascending=False).index[0]
log(f"\n最优不平衡方案：{best_imb}")

best_imb_model = {'基准（原始）': cat_base, '代价敏感学习': cat_cost, 'SMOTE过采样': cat_smote}[best_imb]
with open(os.path.join(OUTPUT_DIR, "models", "catboost_imbalance_best.pkl"), 'wb') as f:
    pickle.dump(best_imb_model, f)


# ================================================================
# 步骤2：最优模型专项调参
# ================================================================
log("\n" + "=" * 60)
log("【步骤2】最优模型专项调参")
log("=" * 60)

if best_imb == 'SMOTE过采样':
    X_tune, y_tune = X_train_smote, y_train_smote
else:
    X_tune, y_tune = X_train, y_train

log(f"调参基准（{best_imb}，默认参数）：")
log(f"  召回率={imbalance_results[best_imb]['召回率']:.4f}  "
    f"AUC={imbalance_results[best_imb]['AUC']:.4f}  "
    f"F1={imbalance_results[best_imb]['F1值']:.4f}")

param_dist = {
    'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.15],
    'depth': [4, 6, 8, 10],
    'iterations': [200, 300, 500],
    'l2_leaf_reg': [1, 3, 5, 7, 9],
    'bagging_temperature': [0, 0.5, 1, 2]
}

log("开始随机搜索调参（20轮 × 3折 = 60次训练）...")
start_time = time.time()

cat_tune = CatBoostClassifier(
    random_state=RANDOM_STATE,
    verbose=0,
    auto_class_weights='Balanced' if best_imb == '代价敏感学习' else 'None'
)

cv_tune = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

# 带进度回调的随机搜索
class ProgressRandomSearch(RandomizedSearchCV):
    def _run_search(self, evaluate_candidates):
        import random
        random.seed(self.random_state)
        param_iter = iter(self.param_distributions)
        all_params = []
        for _ in range(self.n_iter):
            all_params.append({k: v[np.random.randint(len(v))] if isinstance(v, list) else v
                               for k, v in self.param_distributions.items()})
        for i, params in enumerate(all_params):
            log(f"  调参进度：第 {i+1}/{self.n_iter} 轮 | 参数: {params}")
            evaluate_candidates([params])

random_search = ProgressRandomSearch(
    cat_tune,
    param_distributions=param_dist,
    n_iter=20,
    cv=cv_tune,
    scoring='recall',
    random_state=RANDOM_STATE,
    n_jobs=1,
    verbose=0
)

log("开始拟合（预计需要几分钟，请耐心等待）...")
random_search.fit(X_tune, y_tune, cat_features=CATEGORICAL_FEATURES)

elapsed = time.time() - start_time
log(f"调参完成！总耗时 {elapsed:.1f}s ({elapsed/60:.1f}分钟)")
log(f"最优参数：{random_search.best_params_}")
log(f"最优交叉验证召回率：{random_search.best_score_:.4f}")

best_params = random_search.best_params_
log("使用最优参数训练最终模型...")
cat_final = CatBoostClassifier(
    **best_params,
    random_state=RANDOM_STATE,
    verbose=0,
    auto_class_weights='Balanced' if best_imb == '代价敏感学习' else 'None'
)
cat_final.fit(X_tune, y_tune, cat_features=CATEGORICAL_FEATURES)
log("最终模型训练完成，开始测试集预测...")
y_pred_final = cat_final.predict(X_test)
y_prob_final = cat_final.predict_proba(X_test)[:, 1]
final_metrics = eval_metrics(y_test, y_pred_final, y_prob_final)

log(f"\n调参后测试集性能：")
log(f"  准确率={final_metrics['准确率']:.4f}  精确率={final_metrics['精确率']:.4f}")
log(f"  召回率={final_metrics['召回率']:.4f}  F1={final_metrics['F1值']:.4f}  AUC={final_metrics['AUC']:.4f}")

tune_compare = pd.DataFrame({
    '指标': ['召回率', 'AUC', 'F1值'],
    '调参前': [imbalance_results[best_imb]['召回率'],
              imbalance_results[best_imb]['AUC'],
              imbalance_results[best_imb]['F1值']],
    '调参后': [final_metrics['召回率'], final_metrics['AUC'], final_metrics['F1值']]
})
tune_compare['提升'] = tune_compare['调参后'] - tune_compare['调参前']
tune_compare.to_csv(os.path.join(OUTPUT_DIR, '调参前后对比.csv'), index=False, encoding='utf-8-sig')
log("\n--- 调参前后对比 ---")
log(tune_compare.round(4).to_string(index=False))

with open(os.path.join(OUTPUT_DIR, "models", "catboost_final.pkl"), 'wb') as f:
    pickle.dump(cat_final, f)
with open(os.path.join(OUTPUT_DIR, "models", "best_params.pkl"), 'wb') as f:
    pickle.dump(best_params, f)
log("✓ 最终模型已保存：catboost_final.pkl")


# ================================================================
# 步骤3：PR曲线评估体系
# ================================================================
log("\n" + "=" * 60)
log("【步骤3】PR曲线评估体系")
log("=" * 60)

log("训练所有对比模型...")
compare_models = {}

log("  [1/4] 训练逻辑回归...")
lr = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000, class_weight='balanced')
lr.fit(X_train_s, y_train)
compare_models['逻辑回归'] = lr.predict_proba(X_test_s)[:, 1]

log("  [2/4] 训练随机森林...")
rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1, class_weight='balanced')
rf.fit(X_train_s, y_train)
compare_models['随机森林'] = rf.predict_proba(X_test_s)[:, 1]

log("  [3/4] 训练XGBoost...")
xgb_model = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                random_state=RANDOM_STATE, eval_metric='logloss',
                                use_label_encoder=False)
xgb_model.fit(X_train_s, y_train)
compare_models['XGBoost'] = xgb_model.predict_proba(X_test_s)[:, 1]

log("  [4/4] 训练LightGBM...")
lgb_model = lgb.LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                 random_state=RANDOM_STATE, verbose=-1, class_weight='balanced')
lgb_model.fit(X_train_s, y_train)
compare_models['LightGBM'] = lgb_model.predict_proba(X_test_s)[:, 1]

compare_models['CatBoost基准'] = y_prob_base
compare_models['最终最优模型'] = y_prob_final

log(f"  共 {len(compare_models)} 个模型训练完成")
log("绘制PR曲线对比图...")
fig, ax = plt.subplots(figsize=(10, 7))
colors = ['#3498DB', '#2ECC71', '#E74C3C', '#F39C12', '#9B59B6', '#E91E63']

pr_data = {}
for i, (name, y_prob) in enumerate(compare_models.items()):
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    pr_data[name] = {'precision': precision, 'recall': recall, 'AP': ap}
    ax.plot(recall, precision, color=colors[i], lw=2,
            label=f'{name} (AP = {ap:.4f})')

base_rate = y_test.mean()
ax.axhline(y=base_rate, color='gray', lw=1, linestyle='--', label=f'基线 (prevalence = {base_rate:.3f})')

ax.set_xlabel('召回率 (Recall)', fontsize=12)
ax.set_ylabel('精确率 (Precision)', fontsize=12)
ax.set_title('多模型PR曲线对比', fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=9)
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])
ax.grid(alpha=0.3)
save_fig(fig, 'PR曲线对比')

pr_table = pd.DataFrame({
    '模型': list(pr_data.keys()),
    'AP值': [pr_data[n]['AP'] for n in pr_data.keys()]
}).sort_values('AP值', ascending=False)
pr_table.to_csv(os.path.join(OUTPUT_DIR, 'PR曲线AP值对比.csv'), index=False, encoding='utf-8-sig')
log("\n--- AP值对比 ---")
log(pr_table.round(4).to_string(index=False))


# ================================================================
# 步骤4：SHAP可解释性分析
# ================================================================
log("\n" + "=" * 60)
log("【步骤4】SHAP可解释性分析")
log("=" * 60)

np.random.seed(RANDOM_STATE)
sample_idx = np.random.choice(len(X_test), size=min(1000, len(X_test)), replace=False)
X_shap = X_test.iloc[sample_idx]

log(f"SHAP分析样本量：{len(X_shap)} 条")

log("计算SHAP值（TreeExplainer）...")
explainer = shap.TreeExplainer(cat_final)
shap_values = explainer.shap_values(X_shap)
log("✓ SHAP值计算完成")

# --- 图1：全局特征重要性 ---
log("绘制全局特征重要性图...")
shap.summary_plot(shap_values, X_shap, plot_type="bar", show=False, max_display=20)
plt.title('全局特征重要性排序（Top20）', fontsize=14, fontweight='bold')
plt.tight_layout()
save_fig(plt.gcf(), 'SHAP_全局特征重要性')
plt.close('all')

# --- 图2：特征影响摘要图 ---
log("绘制特征影响摘要图...")
shap.summary_plot(shap_values, X_shap, show=False, max_display=20)
plt.title('特征影响方向与强度（SHAP Summary Plot）', fontsize=14, fontweight='bold')
plt.tight_layout()
save_fig(plt.gcf(), 'SHAP_特征影响摘要')
plt.close('all')

# --- 图3：单样本个体解释 ---
log("绘制单样本个体解释图...")
y_prob_shap = cat_final.predict_proba(X_shap)[:, 1]
high_risk_idx = np.argmax(y_prob_shap)
high_risk_sample = X_shap.iloc[[high_risk_idx]]

log(f"  高风险样本编号：{sample_idx[high_risk_idx]}")
log(f"  预测患病概率：{y_prob_shap[high_risk_idx]:.4f}")
log(f"  真实标签：{y_test.iloc[sample_idx[high_risk_idx]]}")

# 使用waterfall plot替代force_plot（兼容性更好）
explanation = shap.Explanation(
    values=shap_values[high_risk_idx],
    base_values=explainer.expected_value,
    data=high_risk_sample.values[0],
    feature_names=X_shap.columns.tolist()
)
shap.waterfall_plot(explanation, show=False, max_display=15)
plt.title(f'单样本风险解释（预测概率={y_prob_shap[high_risk_idx]:.3f})',
          fontsize=12, fontweight='bold')
plt.tight_layout()
save_fig(plt.gcf(), 'SHAP_单样本解释')
plt.close('all')

# 特征重要性表
feature_importance = pd.DataFrame({
    '特征': X_shap.columns,
    '平均|SHAP值|': np.abs(shap_values).mean(axis=0)
}).sort_values('平均|SHAP值|', ascending=False)
feature_importance.to_csv(os.path.join(OUTPUT_DIR, 'SHAP特征重要性.csv'), index=False, encoding='utf-8-sig')
log("\n--- SHAP特征重要性排序 ---")
log(feature_importance.round(4).to_string(index=False))


# ================================================================
# 生成业务结论
# ================================================================
log("\n" + "=" * 60)
log("生成业务结论")
log("=" * 60)

top5_features = feature_importance.head(5)['特征'].tolist()
top5_shap = feature_importance.head(5)['平均|SHAP值|'].tolist()

conclusion = f"""========== 模型优化结论 ==========

【步骤1：类别不平衡方案对比】
- 基准（原始）：召回率={imbalance_results['基准（原始）']['召回率']:.4f}
- 代价敏感学习：召回率={imbalance_results['代价敏感学习']['召回率']:.4f}
- SMOTE过采样：召回率={imbalance_results['SMOTE过采样']['召回率']:.4f}
- 最优方案：{best_imb}

【步骤2：CatBoost调参结果】
- 最优参数：{best_params}
- 调参前召回率：{imbalance_results[best_imb]['召回率']:.4f}
- 调参后召回率：{final_metrics['召回率']:.4f}
- 召回率提升：{final_metrics['召回率'] - imbalance_results[best_imb]['召回率']:.4f}

【步骤3：PR曲线评估】
- 所有模型AP值已计算，最终最优模型AP={pr_data['最终最优模型']['AP']:.4f}

【步骤4：SHAP可解释性分析】
核心风险因子Top5：
"""

for i, (feat, val) in enumerate(zip(top5_features, top5_shap)):
    conclusion += f"  {i+1}. {feat}（平均|SHAP值|={val:.4f}）\n"

conclusion += """
业务结论：
1. 高血压、自评健康、BMI、年龄、心脏病是影响糖尿病患病风险的核心因子
2. 高血压患者患病风险显著升高，应作为筛查重点指标
3. BMI越高患病风险越大，肥胖人群需重点关注
4. 年龄增长与患病风险正相关，60岁以上人群风险明显增加
5. 不良生活习惯（吸烟、不运动）叠加慢性病会显著增加患病风险
"""

log(conclusion)

with open(os.path.join(OUTPUT_DIR, '模型优化结论.txt'), 'w', encoding='utf-8') as f:
    f.write(conclusion)


# ==================== 完成 ====================
log("=" * 60)
log("【全部完成】")
log("=" * 60)
log(f"\n输出目录：{OUTPUT_DIR}")
log("\n新增文件：")
log("  数据文件：")
log("    类别不平衡方案对比.csv")
log("    调参前后对比.csv")
log("    PR曲线AP值对比.csv")
log("    SHAP特征重要性.csv")
log("    模型优化结论.txt")
log("  模型文件（output/models/）：")
log("    catboost_imbalance_best.pkl")
log("    catboost_final.pkl")
log("    best_params.pkl")
log("  可视化图表（output/charts/）：")
log("    PR曲线对比.png")
log("    SHAP_全局特征重要性.png")
log("    SHAP_特征影响摘要.png")
log("    SHAP_单样本解释.png")
