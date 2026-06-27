# -*- coding: utf-8 -*-
"""
糖尿病数据集探索性数据分析（EDA）
数据来源：CDC BRFSS 2015 糖尿病健康指标数据集
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.font_manager import FontProperties
import seaborn as sns
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
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "charts")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 字段中文映射
COL_NAMES_CN = {
    'Diabetes_binary': '糖尿病',
    'HighBP': '高血压',
    'HighChol': '高胆固醇',
    'CholCheck': '胆固醇检查',
    'BMI': 'BMI',
    'Smoker': '吸烟',
    'Stroke': '中风',
    'HeartDiseaseorAttack': '心脏病',
    'PhysActivity': '体育活动',
    'Fruits': '水果摄入',
    'Veggies': '蔬菜摄入',
    'HvyAlcoholConsump': '重度饮酒',
    'AnyHealthcare': '医疗保险',
    'NoDocbcCost': '费用障碍',
    'GenHlth': '自评健康',
    'MentHlth': '心理健康天数',
    'PhysHlth': '身体健康天数',
    'DiffWalk': '行走困难',
    'Sex': '性别',
    'Age': '年龄组',
    'Education': '教育水平',
    'Income': '收入水平'
}

# 年龄组映射
AGE_MAP = {
    1: '18-24', 2: '25-29', 3: '30-34', 4: '35-39',
    5: '40-44', 6: '45-49', 7: '50-54', 8: '55-59',
    9: '60-64', 10: '65-69', 11: '70-74', 12: '75-79', 13: '80+'
}


def save_fig(fig, name):
    """保存图表"""
    path = os.path.join(OUTPUT_DIR, f"{name}.png")
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ 已保存：{name}.png")


# ==================== 1. 加载数据 ====================
print("=" * 60)
print("【第一步】加载清洗后的数据")
print("=" * 60)

data_path = os.path.join(DATA_DIR, "diabetes_cleaned.csv")
df = pd.read_csv(data_path)
print(f"数据形状：{df.shape[0]} 行，{df.shape[1]} 列")

# ==================== 2. 整体描述性统计 ====================
print("\n" + "=" * 60)
print("【第二步】整体描述性统计")
print("=" * 60)

desc_all = df.describe(percentiles=[.25, .5, .75]).T
desc_all['中位数'] = df.median()
desc_all = desc_all[['count', 'mean', '中位数', 'std', 'min', '25%', '50%', '75%', 'max']]
desc_all.columns = ['样本量', '均值', '中位数', '标准差', '最小值', '25%分位', '50%分位', '75%分位', '最大值']
desc_all.index = [COL_NAMES_CN.get(col, col) for col in desc_all.index]
desc_all.to_csv(os.path.join(OUTPUT_DIR, '..', 'descriptive_stats.csv'), encoding='utf-8-sig')
print(desc_all.to_string())

print("\n--- 正负样本占比 ---")
total = len(df)
positive = df['Diabetes_binary'].sum()
negative = total - positive
print(f"总样本量：{total}")
print(f"糖尿病患者（正样本）：{int(positive)} 人，占比 {positive/total*100:.2f}%")
print(f"未患病（负样本）：{int(negative)} 人，占比 {negative/total*100:.2f}%")

# ==================== 3. 单维度特征与患病关联分析 ====================
print("\n" + "=" * 60)
print("【第三步】单维度特征与患病关联分析")
print("=" * 60)

# --- 图1：年龄分层患病率 ---
print("\n--- 绘制年龄分层患病率 ---")
df['Age_group'] = df['Age'].map(AGE_MAP)
age_order = list(AGE_MAP.values())
age_diag = df.groupby('Age_group')['Diabetes_binary'].mean() * 100
age_diag = age_diag.reindex(age_order)

fig, ax = plt.subplots(figsize=(12, 6))
colors = plt.cm.RdYlGn_r(age_diag.values / age_diag.values.max())
bars = ax.bar(range(len(age_diag)), age_diag.values, color=colors)
ax.set_xticks(range(len(age_diag)))
ax.set_xticklabels(age_diag.index, rotation=45, ha='right')
ax.set_xlabel('年龄组', fontsize=12)
ax.set_ylabel('糖尿病患病率 (%)', fontsize=12)
ax.set_title('各年龄组糖尿病患病率', fontsize=14, fontweight='bold')
for i, v in enumerate(age_diag.values):
    ax.text(i, v + 0.3, f'{v:.1f}%', ha='center', fontsize=9)
ax.set_ylim(0, age_diag.max() * 1.15)
ax.grid(axis='y', alpha=0.3)
save_fig(fig, '01_年龄分层患病率')

# --- 图2：性别患病率对比 ---
print("\n--- 绘制性别患病率对比 ---")
sex_diag = df.groupby('Sex')['Diabetes_binary'].mean() * 100

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(['女性', '男性'], sex_diag.values, color=['#FF6B6B', '#4ECDC4'], width=0.5)
ax.set_ylabel('糖尿病患病率 (%)', fontsize=12)
ax.set_title('不同性别的糖尿病患病率', fontsize=14, fontweight='bold')
for bar, v in zip(bars, sex_diag.values):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.3, f'{v:.2f}%', ha='center', fontsize=11, fontweight='bold')
ax.set_ylim(0, sex_diag.max() * 1.2)
ax.grid(axis='y', alpha=0.3)
save_fig(fig, '02_性别患病率对比')

# --- 图3：BMI分布直方图 ---
print("\n--- 绘制BMI分布直方图 ---")
fig, ax = plt.subplots(figsize=(12, 6))
ax.hist(df[df['Diabetes_binary']==0]['BMI'], bins=50, alpha=0.6, label='未患病', color='#2ECC71', density=True)
ax.hist(df[df['Diabetes_binary']==1]['BMI'], bins=50, alpha=0.6, label='患病', color='#E74C3C', density=True)
ax.axvline(x=30, color='orange', linestyle='--', alpha=0.7, label='肥胖线(BMI=30)')
ax.set_xlabel('BMI', fontsize=12)
ax.set_ylabel('密度', fontsize=12)
ax.set_title('BMI分布（按是否患病分组）', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)
save_fig(fig, '03_BMI分布直方图')

# --- 图4：高血压/高胆固醇患病率 ---
print("\n--- 绘制高血压/高胆固醇患病率 ---")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

bp_diag = df.groupby('HighBP')['Diabetes_binary'].mean() * 100
bars1 = axes[0].bar(['无高血压', '有高血压'], bp_diag.values, color=['#3498DB', '#E74C3C'], width=0.5)
axes[0].set_ylabel('患病率 (%)', fontsize=11)
axes[0].set_title('高血压与糖尿病患病率', fontsize=13, fontweight='bold')
for bar, v in zip(bars1, bp_diag.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, v + 0.3, f'{v:.2f}%', ha='center', fontsize=10)
axes[0].set_ylim(0, bp_diag.max() * 1.25)
axes[0].grid(axis='y', alpha=0.3)

chol_diag = df.groupby('HighChol')['Diabetes_binary'].mean() * 100
bars2 = axes[1].bar(['无高胆固醇', '有高胆固醇'], chol_diag.values, color=['#3498DB', '#E67E22'], width=0.5)
axes[1].set_ylabel('患病率 (%)', fontsize=11)
axes[1].set_title('高胆固醇与糖尿病患病率', fontsize=13, fontweight='bold')
for bar, v in zip(bars2, chol_diag.values):
    axes[1].text(bar.get_x() + bar.get_width()/2, v + 0.3, f'{v:.2f}%', ha='center', fontsize=10)
axes[1].set_ylim(0, chol_diag.max() * 1.25)
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
save_fig(fig, '04_高血压高胆固醇患病率')

# --- 图5：吸烟患病率对比 ---
print("\n--- 绘制吸烟患病率对比 ---")
smoke_diag = df.groupby('Smoker')['Diabetes_binary'].mean() * 100

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(['不吸烟', '吸烟'], smoke_diag.values, color=['#2ECC71', '#95A5A6'], width=0.5)
ax.set_ylabel('患病率 (%)', fontsize=12)
ax.set_title('吸烟与糖尿病患病率', fontsize=14, fontweight='bold')
for bar, v in zip(bars, smoke_diag.values):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.3, f'{v:.2f}%', ha='center', fontsize=11, fontweight='bold')
ax.set_ylim(0, smoke_diag.max() * 1.25)
ax.grid(axis='y', alpha=0.3)
save_fig(fig, '05_吸烟患病率对比')

# --- 图6：体育活动患病率 ---
print("\n--- 绘制体育活动患病率 ---")
phys_diag = df.groupby('PhysActivity')['Diabetes_binary'].mean() * 100

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(['无运动', '有运动'], phys_diag.values, color=['#E74C3C', '#2ECC71'], width=0.5)
ax.set_ylabel('患病率 (%)', fontsize=12)
ax.set_title('体育活动与糖尿病患病率', fontsize=14, fontweight='bold')
for bar, v in zip(bars, phys_diag.values):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.3, f'{v:.2f}%', ha='center', fontsize=11, fontweight='bold')
ax.set_ylim(0, phys_diag.max() * 1.25)
ax.grid(axis='y', alpha=0.3)
save_fig(fig, '06_体育活动患病率')

# --- 图7：蔬果摄入患病率 ---
print("\n--- 绘制蔬果摄入患病率 ---")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

fruit_diag = df.groupby('Fruits')['Diabetes_binary'].mean() * 100
bars1 = axes[0].bar(['不摄入水果', '摄入水果'], fruit_diag.values, color=['#E67E22', '#2ECC71'], width=0.5)
axes[0].set_ylabel('患病率 (%)', fontsize=11)
axes[0].set_title('水果摄入与糖尿病患病率', fontsize=13, fontweight='bold')
for bar, v in zip(bars1, fruit_diag.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, v + 0.3, f'{v:.2f}%', ha='center', fontsize=10)
axes[0].set_ylim(0, fruit_diag.max() * 1.25)
axes[0].grid(axis='y', alpha=0.3)

veg_diag = df.groupby('Veggies')['Diabetes_binary'].mean() * 100
bars2 = axes[1].bar(['不摄入蔬菜', '摄入蔬菜'], veg_diag.values, color=['#E67E22', '#27AE60'], width=0.5)
axes[1].set_ylabel('患病率 (%)', fontsize=11)
axes[1].set_title('蔬菜摄入与糖尿病患病率', fontsize=13, fontweight='bold')
for bar, v in zip(bars2, veg_diag.values):
    axes[1].text(bar.get_x() + bar.get_width()/2, v + 0.3, f'{v:.2f}%', ha='center', fontsize=10)
axes[1].set_ylim(0, veg_diag.max() * 1.25)
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
save_fig(fig, '07_蔬果摄入患病率')

# ==================== 4. 多维度交叉分析 ====================
print("\n" + "=" * 60)
print("【第四步】多维度交叉分析")
print("=" * 60)

# --- 图8：BMI肥胖 + 高血压 ---
print("\n--- 绘制BMI+高血压交叉分析 ---")
df['肥胖'] = (df['BMI'] >= 30).astype(int)
cross1 = df.groupby(['肥胖', 'HighBP'])['Diabetes_binary'].mean() * 100
cross1.index = ['非肥胖+无高血压', '非肥胖+有高血压', '肥胖+无高血压', '肥胖+有高血压']

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(range(len(cross1)), cross1.values, color=['#2ECC71', '#F39C12', '#E67E22', '#E74C3C'])
ax.set_xticks(range(len(cross1)))
ax.set_xticklabels(cross1.index, fontsize=10)
ax.set_ylabel('患病率 (%)', fontsize=12)
ax.set_title('BMI肥胖与高血压交叉分析', fontsize=14, fontweight='bold')
for i, v in enumerate(cross1.values):
    ax.text(i, v + 0.5, f'{v:.2f}%', ha='center', fontsize=11, fontweight='bold')
ax.set_ylim(0, cross1.max() * 1.2)
ax.grid(axis='y', alpha=0.3)
save_fig(fig, '08_BMI高血压交叉分析')

# --- 图9：高龄 + 吸烟 ---
print("\n--- 绘制高龄+吸烟交叉分析 ---")
df['高龄'] = (df['Age'] >= 9).astype(int)
cross2 = df.groupby(['高龄', 'Smoker'])['Diabetes_binary'].mean() * 100
cross2.index = ['非高龄+不吸烟', '非高龄+吸烟', '高龄+不吸烟', '高龄+吸烟']

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(range(len(cross2)), cross2.values, color=['#2ECC71', '#3498DB', '#E67E22', '#E74C3C'])
ax.set_xticks(range(len(cross2)))
ax.set_xticklabels(cross2.index, fontsize=10)
ax.set_ylabel('患病率 (%)', fontsize=12)
ax.set_title('高龄与吸烟交叉分析', fontsize=14, fontweight='bold')
for i, v in enumerate(cross2.values):
    ax.text(i, v + 0.5, f'{v:.2f}%', ha='center', fontsize=11, fontweight='bold')
ax.set_ylim(0, cross2.max() * 1.2)
ax.grid(axis='y', alpha=0.3)
save_fig(fig, '09_高龄吸烟交叉分析')

# --- 图10：运动 + 肥胖 ---
print("\n--- 绘制运动+肥胖交叉分析 ---")
cross3 = df.groupby(['PhysActivity', '肥胖'])['Diabetes_binary'].mean() * 100
cross3.index = ['无运动+非肥胖', '有运动+非肥胖', '无运动+肥胖', '有运动+肥胖']

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(range(len(cross3)), cross3.values, color=['#2ECC71', '#3498DB', '#E67E22', '#E74C3C'])
ax.set_xticks(range(len(cross3)))
ax.set_xticklabels(cross3.index, fontsize=10)
ax.set_ylabel('患病率 (%)', fontsize=12)
ax.set_title('运动习惯与肥胖交叉分析', fontsize=14, fontweight='bold')
for i, v in enumerate(cross3.values):
    ax.text(i, v + 0.5, f'{v:.2f}%', ha='center', fontsize=11, fontweight='bold')
ax.set_ylim(0, cross3.max() * 1.2)
ax.grid(axis='y', alpha=0.3)
save_fig(fig, '10_运动肥胖交叉分析')

# ==================== 5. 全局相关性分析 ====================
print("\n" + "=" * 60)
print("【第五步】全局相关性分析")
print("=" * 60)

# --- 图11：相关性热力图 ---
print("\n--- 绘制相关性热力图 ---")
corr_cols = [c for c in df.columns if c not in ['Age_group', '肥胖', '高龄']]
corr_matrix = df[corr_cols].corr()

corr_cn = corr_matrix.copy()
corr_cn.index = [COL_NAMES_CN.get(c, c) for c in corr_cn.index]
corr_cn.columns = [COL_NAMES_CN.get(c, c) for c in corr_cn.columns]

fig, ax = plt.subplots(figsize=(16, 14))
mask = np.triu(np.ones_like(corr_cn, dtype=bool), k=1)
sns.heatmap(corr_cn, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, vmin=-1, vmax=1, square=True, ax=ax,
            linewidths=0.5, cbar_kws={"shrink": 0.8},
            annot_kws={"size": 8})
ax.set_title('特征相关性热力图', fontsize=16, fontweight='bold', pad=20)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=10)
save_fig(fig, '11_相关性热力图')

# --- 图12：Top10核心影响特征 ---
print("\n--- 绘制Top10核心特征 ---")
target_corr = corr_matrix['Diabetes_binary'].drop('Diabetes_binary').abs().sort_values(ascending=False)
target_corr_cn = target_corr.copy()
target_corr_cn.index = [COL_NAMES_CN.get(c, c) for c in target_corr_cn.index]

fig, ax = plt.subplots(figsize=(10, 7))
top10 = target_corr_cn.head(10)
colors = plt.cm.RdYlGn_r(np.linspace(0.3, 0.9, len(top10)))
bars = ax.barh(range(len(top10)), top10.values, color=colors)
ax.set_yticks(range(len(top10)))
ax.set_yticklabels(top10.index, fontsize=11)
ax.set_xlabel('相关系数（绝对值）', fontsize=12)
ax.set_title('与糖尿病相关性Top10特征', fontsize=14, fontweight='bold')
for i, v in enumerate(top10.values):
    ax.text(v + 0.003, i, f'{v:.4f}', va='center', fontsize=10)
ax.set_xlim(0, top10.max() * 1.15)
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3)
save_fig(fig, '12_Top10相关特征')

# ==================== 6. 输出分析结论 ====================
print("\n" + "=" * 60)
print("【第六步】分析结论汇总")
print("=" * 60)

top5_text = '\n'.join([
    f"  {i+1}. {name}（相关系数：{val:.4f}）"
    for i, (name, val) in enumerate(target_corr_cn.head(5).items())
])

conclusion = f"""
========== 探索性数据分析结论 ==========

【数据概况】
- 总样本量：{len(df)} 条
- 特征数量：{len([c for c in df.columns if c not in ['Age_group', '肥胖', '高龄']])} 个
- 基线患病率：{df['Diabetes_binary'].mean()*100:.2f}%

【单因素分析关键发现】
1. 年龄：患病率随年龄增长显著上升，60岁以上人群风险明显增加
2. 性别：男性患病率略高于女性
3. BMI：糖尿病患者BMI分布整体偏高，肥胖人群患病风险显著增加
4. 高血压：有高血压人群患病率远高于无高血压人群
5. 高胆固醇：有高胆固醇人群患病率高于无高胆固醇人群
6. 吸烟：吸烟人群患病率高于不吸烟人群
7. 体育活动：不运动人群患病率显著高于运动人群

【多因素交叉分析关键发现】
1. BMI肥胖+高血压：两者叠加后患病率最高，呈现显著的协同效应
2. 高龄+吸烟：高龄吸烟者的患病风险远高于其他组合
3. 运动+肥胖：即使肥胖，保持运动也能在一定程度上降低患病风险

【核心影响特征Top5】
{top5_text}

【建模建议】
- 高血压、自评健康、BMI、年龄、心脏病是最具预测价值的特征
- 类别不平衡问题需在建模阶段处理（如SMOTE、代价敏感学习等）
"""

print(conclusion)

conclusion_path = os.path.join(OUTPUT_DIR, '..', 'eda_conclusions.txt')
with open(conclusion_path, 'w', encoding='utf-8') as f:
    f.write(conclusion)
print("✓ 分析结论已保存")

# ==================== 完成 ====================
print("\n" + "=" * 60)
print("【全部完成】")
print("=" * 60)
print(f"\n所有图表已保存至：{OUTPUT_DIR}")
print("\n输出文件列表：")
print("  descriptive_stats.csv — 描述性统计结果表")
print("  eda_conclusions.txt — 分析结论")
print("  01~12_*.png — 12张分析图表")
