# -*- coding: utf-8 -*-
"""
糖尿病预测模型训练与评估
模型：逻辑回归、随机森林、XGBoost、LightGBM、CatBoost、决策树、SVM
"""

import pandas as pd
import numpy as np
import os
import pickle
import time
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.font_manager import FontProperties
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import (confusion_matrix, roc_curve, auc,
                             accuracy_score, precision_score, recall_score, f1_score)
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

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
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
CHART_DIR = os.path.join(OUTPUT_DIR, "charts")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CHART_DIR, exist_ok=True)

RANDOM_STATE = 42
CV_FOLDS = 5

# 类别特征列名（用于CatBoost）
CATEGORICAL_FEATURES = ['HighBP', 'HighChol', 'CholCheck', 'Smoker', 'Stroke',
                        'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
                        'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost',
                        'GenHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income']


def save_fig(fig, name):
    """保存图表"""
    path = os.path.join(CHART_DIR, f"{name}.png")
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ 已保存：{name}.png")


def evaluate_model(y_true, y_pred, y_prob):
    """计算评估指标"""
    return {
        '准确率': accuracy_score(y_true, y_pred),
        '精确率': precision_score(y_true, y_pred),
        '召回率': recall_score(y_true, y_pred),
        'F1值': f1_score(y_true, y_pred),
        'AUC': auc(*roc_curve(y_true, y_prob)[:2])
    }


# ==================== 1. 加载数据 ====================
print("=" * 60)
print("【第一步】加载数据")
print("=" * 60)

# 标准化数据集（逻辑回归、随机森林、XGBoost、LightGBM）
train_scaled = pd.read_csv(os.path.join(DATA_DIR, "train_scaled.csv"))
test_scaled = pd.read_csv(os.path.join(DATA_DIR, "test_scaled.csv"))

X_train = train_scaled.drop(columns=['Diabetes_binary'])
y_train = train_scaled['Diabetes_binary']
X_test = test_scaled.drop(columns=['Diabetes_binary'])
y_test = test_scaled['Diabetes_binary']

# CatBoost专用数据集
train_cat = pd.read_csv(os.path.join(DATA_DIR, "train_catboost.csv"))
test_cat = pd.read_csv(os.path.join(DATA_DIR, "test_catboost.csv"))

X_train_cat = train_cat.drop(columns=['Diabetes_binary'])
y_train_cat = train_cat['Diabetes_binary']
X_test_cat = test_cat.drop(columns=['Diabetes_binary'])
y_test_cat = test_cat['Diabetes_binary']

print(f"训练集：{X_train.shape[0]} 条，测试集：{X_test.shape[0]} 条")
print(f"特征数量：{X_train.shape[1]} 个")

# ==================== 2. 定义模型 ====================
print("\n" + "=" * 60)
print("【第二步】定义模型")
print("=" * 60)

models = {
    '逻辑回归': LogisticRegression(
        random_state=RANDOM_STATE,
        max_iter=1000,
        class_weight='balanced'
    ),
    '随机森林': RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight='balanced'
    ),
    'XGBoost': xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
        eval_metric='logloss',
        use_label_encoder=False,
        scale_pos_weight=(len(y_train) - y_train.sum()) / y_train.sum()
    ),
    'LightGBM': lgb.LGBMClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
        verbose=-1,
        class_weight='balanced'
    ),
    'CatBoost': CatBoostClassifier(
        iterations=200,
        depth=6,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
        verbose=0,
        auto_class_weights='Balanced'
    ),
    '决策树': DecisionTreeClassifier(
        max_depth=10,
        random_state=RANDOM_STATE,
        class_weight='balanced'
    ),
    'SVM': CalibratedClassifierCV(
        LinearSVC(
            max_iter=2000,
            random_state=RANDOM_STATE,
            class_weight='balanced'
        ),
        cv=3  # 使用线性核，受大样本算力限制，非线性核在十万级以上样本中训练效率极低
    )
}

for name, model in models.items():
    print(f"  {name}: {model.__class__.__name__}")

# ==================== 3. 5折分层交叉验证 ====================
print("\n" + "=" * 60)
print("【第三步】5折分层交叉验证")
print("=" * 60)

cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

cv_results = {}
for name, model in models.items():
    print(f"\n训练 {name}...")
    start_time = time.time()

    if name == 'CatBoost':
        cv_res = cross_validate(model, X_train_cat, y_train_cat, cv=cv,
                                scoring=scoring, return_train_score=False)
    else:
        cv_res = cross_validate(model, X_train, y_train, cv=cv,
                                scoring=scoring, return_train_score=False)

    elapsed = time.time() - start_time

    cv_results[name] = {
        '准确率': f"{cv_res['test_accuracy'].mean():.4f} ± {cv_res['test_accuracy'].std():.4f}",
        '精确率': f"{cv_res['test_precision'].mean():.4f} ± {cv_res['test_precision'].std():.4f}",
        '召回率': f"{cv_res['test_recall'].mean():.4f} ± {cv_res['test_recall'].std():.4f}",
        'F1值': f"{cv_res['test_f1'].mean():.4f} ± {cv_res['test_f1'].std():.4f}",
        'AUC': f"{cv_res['test_roc_auc'].mean():.4f} ± {cv_res['test_roc_auc'].std():.4f}",
        '耗时(s)': f"{elapsed:.2f}"
    }
    print(f"  ✓ {name} 完成，耗时 {elapsed:.2f}s")
    print(f"    AUC: {cv_results[name]['AUC']}")
    print(f"    召回率: {cv_results[name]['召回率']}")

cv_df = pd.DataFrame(cv_results).T
cv_df.to_csv(os.path.join(OUTPUT_DIR, '交叉验证结果.csv'), encoding='utf-8-sig')
print("\n--- 交叉验证结果汇总 ---")
print(cv_df.to_string())

# ==================== 4. 训练模型并评估测试集 ====================
print("\n" + "=" * 60)
print("【第四步】训练模型并评估测试集")
print("=" * 60)

test_results = {}
trained_models = {}
y_probs = {}

for name, model in models.items():
    print(f"\n训练 {name}...")
    start_time = time.time()

    if name == 'CatBoost':
        model.fit(X_train_cat, y_train_cat, cat_features=CATEGORICAL_FEATURES)
        y_pred = model.predict(X_test_cat)
        y_prob = model.predict_proba(X_test_cat)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

    elapsed = time.time() - start_time
    metrics = evaluate_model(y_test, y_pred, y_prob)

    test_results[name] = metrics
    trained_models[name] = model
    y_probs[name] = y_prob

    model_path = os.path.join(MODEL_DIR, f"{name}_model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    print(f"  ✓ {name} 完成，耗时 {elapsed:.2f}s")
    print(f"    准确率={metrics['准确率']:.4f}  精确率={metrics['精确率']:.4f}  "
          f"召回率={metrics['召回率']:.4f}  F1={metrics['F1值']:.4f}  AUC={metrics['AUC']:.4f}")

test_df = pd.DataFrame(test_results).T
test_df.to_csv(os.path.join(OUTPUT_DIR, '测试集评估结果.csv'), encoding='utf-8-sig')
print("\n--- 测试集评估结果汇总 ---")
print(test_df.round(4).to_string())

# ==================== 5. 绘制评估可视化 ====================
print("\n" + "=" * 60)
print("【第五步】绘制评估可视化")
print("=" * 60)

model_names = list(models.keys())
colors = ['#3498DB', '#2ECC71', '#E74C3C', '#F39C12', '#9B59B6', '#1ABC9C', '#E91E63']

# --- 混淆矩阵热力图 ---
print("\n--- 绘制混淆矩阵 ---")
for i, name in enumerate(model_names):
    if name == 'CatBoost':
        y_pred = trained_models[name].predict(X_test_cat)
    else:
        y_pred = trained_models[name].predict(X_test)

    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=[0, 1], yticks=[0, 1],
           xticklabels=['未患病', '患病'],
           yticklabels=['未患病', '患病'],
           ylabel='真实标签', xlabel='预测标签',
           title=f'{name} — 混淆矩阵')

    thresh = cm.max() / 2
    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            ax.text(col, row, format(cm[row, col], 'd'),
                    ha="center", va="center",
                    color="white" if cm[row, col] > thresh else "black",
                    fontsize=14, fontweight='bold')

    ax.grid(False)
    save_fig(fig, f'混淆矩阵_{name}')

# --- ROC对比曲线 ---
print("\n--- 绘制ROC对比曲线 ---")
fig, ax = plt.subplots(figsize=(9, 7))
for i, name in enumerate(model_names):
    fpr, tpr, _ = roc_curve(y_test, y_probs[name])
    roc_auc_val = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=colors[i], lw=2,
            label=f'{name} (AUC = {roc_auc_val:.4f})')

ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='随机猜测')
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])
ax.set_xlabel('假正率 (FPR)', fontsize=12)
ax.set_ylabel('真正率 (TPR)', fontsize=12)
ax.set_title('7个模型ROC曲线对比', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.grid(alpha=0.3)
save_fig(fig, 'ROC曲线对比')

# --- 核心指标横向对比 ---
print("\n--- 绘制核心指标对比柱状图 ---")
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

metrics_to_plot = ['召回率', 'AUC', 'F1值']
for idx, metric in enumerate(metrics_to_plot):
    vals = [test_results[name][metric] for name in model_names]
    bars = axes[idx].bar(range(len(model_names)), vals, color=colors, width=0.6)
    axes[idx].set_xticks(range(len(model_names)))
    axes[idx].set_xticklabels(model_names, fontsize=10, rotation=15)
    axes[idx].set_title(metric, fontsize=13, fontweight='bold')
    axes[idx].set_ylim(0, max(vals) * 1.15)
    axes[idx].grid(axis='y', alpha=0.3)
    for bar, v in zip(bars, vals):
        axes[idx].text(bar.get_x() + bar.get_width()/2, v + 0.005,
                       f'{v:.4f}', ha='center', fontsize=9, fontweight='bold')

plt.suptitle('7个模型核心指标横向对比', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
save_fig(fig, '核心指标横向对比')

# ==================== 6. 选出最优模型 ====================
print("\n" + "=" * 60)
print("【第六步】选出最优模型")
print("=" * 60)

auc_rank = test_df.sort_values('AUC', ascending=False)
print("\n--- 按AUC排序 ---")
print(auc_rank[['AUC', '召回率', 'F1值']].round(4).to_string())

recall_rank = test_df.sort_values('召回率', ascending=False)
print("\n--- 按召回率排序 ---")
print(recall_rank[['召回率', 'AUC', 'F1值']].round(4).to_string())

best_auc = auc_rank.index[0]
best_recall = recall_rank.index[0]

# 综合评分：AUC*0.5 + 召回率*0.3 + F1*0.2
test_df['综合评分'] = (test_df['AUC'] * 0.5 + test_df['召回率'] * 0.3 + test_df['F1值'] * 0.2)
overall_rank = test_df.sort_values('综合评分', ascending=False)
best_overall = overall_rank.index[0]

print(f"\n--- 最优模型 ---")
print(f"AUC最优：{best_auc}（AUC={test_results[best_auc]['AUC']:.4f}）")
print(f"召回率最优：{best_recall}（召回率={test_results[best_recall]['召回率']:.4f}）")
print(f"综合最优：{best_overall}（综合评分={overall_rank.loc[best_overall, '综合评分']:.4f}）")

overall_rank.to_csv(os.path.join(OUTPUT_DIR, '模型综合排名.csv'), encoding='utf-8-sig')

# ==================== 完成 ====================
print("\n" + "=" * 60)
print("【全部完成】")
print("=" * 60)
print(f"\n输出目录：{OUTPUT_DIR}")
print("\n输出文件列表：")
print("  数据文件：")
print("    1. 交叉验证结果.csv")
print("    2. 测试集评估结果.csv")
print("    3. 模型综合排名.csv")
print("\n  模型文件：")
for name in model_names:
    print(f"    {name}_model.pkl")
print("\n  可视化图表（output/charts/）：")
print("    混淆矩阵 × 7张")
print("    ROC曲线对比 × 1张")
print("    核心指标横向对比 × 1张")
