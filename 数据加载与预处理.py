# -*- coding: utf-8 -*-
"""
糖尿病数据集加载与预处理脚本
数据来源：CDC BRFSS 2015 糖尿病健康指标数据集
"""

import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ==================== 配置参数 ====================
# 项目根目录（脚本所在目录）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "原始数据集")
OUTPUT_DIR = BASE_DIR

# 数据文件名
DATA_FILE = "diabetes_binary_health_indicators_BRFSS2015.csv"

# 随机种子
RANDOM_STATE = 42

# 测试集比例
TEST_SIZE = 0.3

# ==================== 1. 加载数据 ====================
print("=" * 60)
print("【第一步】加载数据")
print("=" * 60)

data_path = os.path.join(DATA_DIR, DATA_FILE)
df = pd.read_csv(data_path)

print(f"数据文件：{DATA_FILE}")
print(f"数据形状：{df.shape[0]} 行，{df.shape[1]} 列")
print(f"\n前5行数据预览：")
print(df.head())
print(f"\n数据类型：")
print(df.dtypes)

# ==================== 2. 数据质量探查 ====================
print("\n" + "=" * 60)
print("【第二步】数据质量探查")
print("=" * 60)

# 2.1 缺失值统计
print("\n--- 缺失值统计 ---")
missing_values = df.isnull().sum()
missing_percent = (missing_values / len(df)) * 100
missing_df = pd.DataFrame({
    '缺失数量': missing_values,
    '缺失占比(%)': missing_percent
})
print(missing_df[missing_df['缺失数量'] > 0])
if missing_values.sum() == 0:
    print("✓ 无缺失值")
else:
    print(f"共 {missing_values.sum()} 个缺失值，占总数据的 {missing_percent.mean():.2f}%")

# 2.2 重复值统计
print("\n--- 重复值统计 ---")
duplicate_count = df.duplicated().sum()
duplicate_percent = (duplicate_count / len(df)) * 100
print(f"重复行数量：{duplicate_count}")
print(f"重复行占比：{duplicate_percent:.2f}%")

# 2.3 各字段取值范围
print("\n--- 各字段取值范围 ---")
for col in df.columns:
    print(f"{col:25s} 最小值: {df[col].min():8.2f}  最大值: {df[col].max():8.2f}  "
          f"均值: {df[col].mean():8.2f}  标准差: {df[col].std():8.2f}")

# 2.4 逻辑异常值识别
print("\n--- 逻辑异常值识别 ---")

# BMI异常值：正常范围通常为15-60，超出视为异常
bmi_abnormal = df[(df['BMI'] < 15) | (df['BMI'] > 60)]
print(f"BMI异常值（<15 或 >60）：{len(bmi_abnormal)} 条，占比 {len(bmi_abnormal)/len(df)*100:.2f}%")

# 年龄异常值：应为1-13的整数
age_abnormal = df[(df['Age'] < 1) | (df['Age'] > 13)]
print(f"年龄异常值（<1 或 >13）：{len(age_abnormal)} 条，占比 {len(age_abnormal)/len(df)*100:.2f}%")

# 二分类变量异常值：应为0或1
binary_cols = ['HighBP', 'HighChol', 'CholCheck', 'Smoker', 'Stroke',
               'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
               'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'DiffWalk',
               'Sex', 'Diabetes_binary']
for col in binary_cols:
    abnormal = df[~df[col].isin([0, 1])]
    if len(abnormal) > 0:
        print(f"{col} 异常值：{len(abnormal)} 条")

# 2.5 目标变量分布
print("\n--- 目标变量分布 ---")
print(df['Diabetes_binary'].value_counts())
print(f"\n正样本（糖尿病）占比：{df['Diabetes_binary'].mean()*100:.2f}%")

# ==================== 3. 全流程数据清洗 ====================
print("\n" + "=" * 60)
print("【第三步】数据清洗")
print("=" * 60)

df_clean = df.copy()

# 3.1 删除重复行
print("\n--- 删除重复行 ---")
before_count = len(df_clean)
df_clean = df_clean.drop_duplicates()
after_count = len(df_clean)
print(f"删除前：{before_count} 行")
print(f"删除后：{after_count} 行")
print(f"删除重复行：{before_count - after_count} 行")

# 3.2 缺失值填充（众数填充）
print("\n--- 缺失值填充（众数） ---")
missing_cols = df_clean.columns[df_clean.isnull().any()].tolist()
if missing_cols:
    for col in missing_cols:
        mode_value = df_clean[col].mode()[0]
        fill_count = df_clean[col].isnull().sum()
        df_clean[col] = df_clean[col].fillna(mode_value)
        print(f"{col}：填充 {fill_count} 个缺失值，使用众数 {mode_value}")
else:
    print("✓ 无缺失值需要填充")

# 3.3 异常值处理（盖帽法）
print("\n--- 异常值处理（盖帽法） ---")

# BMI盖帽法：使用1%和99%分位数
bmi_lower = df_clean['BMI'].quantile(0.01)
bmi_upper = df_clean['BMI'].quantile(0.99)
bmi_before = df_clean['BMI'].copy()
df_clean['BMI'] = df_clean['BMI'].clip(lower=bmi_lower, upper=bmi_upper)
bmi修正数量 = int(((bmi_before != df_clean['BMI']) & ((bmi_before < bmi_lower) | (bmi_before > bmi_upper))).sum())
print(f"BMI盖帽范围：[{bmi_lower:.1f}, {bmi_upper:.1f}]")
print(f"修正异常值：{bmi修正数量} 条")

# MentHlth盖帽法：0-30范围外的修正
ment_before = df_clean['MentHlth'].copy()
df_clean['MentHlth'] = df_clean['MentHlth'].clip(lower=0, upper=30)
ment修正数量 = int(((ment_before != df_clean['MentHlth']) & ((ment_before < 0) | (ment_before > 30))).sum())
print(f"MentHlth修正范围：[0, 30]")
print(f"修正异常值：{ment修正数量} 条")

# PhysHlth盖帽法：0-30范围外的修正
phys_before = df_clean['PhysHlth'].copy()
df_clean['PhysHlth'] = df_clean['PhysHlth'].clip(lower=0, upper=30)
phys修正数量 = int(((phys_before != df_clean['PhysHlth']) & ((phys_before < 0) | (phys_before > 30))).sum())
print(f"PhysHlth修正范围：[0, 30]")
print(f"修正异常值：{phys修正数量} 条")

# 3.4 格式统一
print("\n--- 格式统一 ---")

# 二分类变量确保为整数
binary_cols_clean = ['HighBP', 'HighChol', 'CholCheck', 'Smoker', 'Stroke',
                     'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
                     'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'DiffWalk',
                     'Sex', 'Diabetes_binary']
for col in binary_cols_clean:
    df_clean[col] = df_clean[col].astype(int)
print(f"二分类变量（{len(binary_cols_clean)}个）已转换为整数类型")

# 序数分类变量确保为整数
ordinal_cols = ['GenHlth', 'Age', 'Education', 'Income']
for col in ordinal_cols:
    df_clean[col] = df_clean[col].astype(int)
print(f"序数分类变量（{len(ordinal_cols)}个）已转换为整数类型")

# 连续变量确保为浮点数
continuous_cols = ['BMI', 'MentHlth', 'PhysHlth']
for col in continuous_cols:
    df_clean[col] = df_clean[col].astype(float)
print(f"连续变量（{len(continuous_cols)}个）已转换为浮点数类型")

print(f"\n清洗后数据形状：{df_clean.shape[0]} 行，{df_clean.shape[1]} 列")

# ==================== 4. 特征工程预处理 ====================
print("\n" + "=" * 60)
print("【第四步】特征工程预处理")
print("=" * 60)

# 4.1 区分特征类型
print("\n--- 特征类型划分 ---")

# 类别特征（保留数值编码，用于逻辑回归、随机森林、XGBoost、LightGBM）
categorical_features = ['HighBP', 'HighChol', 'CholCheck', 'Smoker', 'Stroke',
                        'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
                        'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost',
                        'GenHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income']

# 连续特征
continuous_features = ['BMI', 'MentHlth', 'PhysHlth']

# 目标变量
target = 'Diabetes_binary'

print(f"类别特征（{len(categorical_features)}个）：{categorical_features}")
print(f"连续特征（{len(continuous_features)}个）：{continuous_features}")
print(f"目标变量：{target}")

# 4.2 特征与目标变量分离
X = df_clean.drop(columns=[target])
y = df_clean[target]

# ==================== 5. 数据集分层划分 ====================
print("\n" + "=" * 60)
print("【第五步】数据集分层划分")
print("=" * 60)

# 5.1 分层抽样划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y  # 分层抽样
)

print(f"划分比例：训练集 {int((1-TEST_SIZE)*100)}%，测试集 {int(TEST_SIZE*100)}%")
print(f"随机种子：{RANDOM_STATE}")
print(f"\n训练集：{X_train.shape[0]} 条")
print(f"  - 正样本：{int(y_train.sum())} 条 ({y_train.mean()*100:.2f}%)")
print(f"  - 负样本：{int(len(y_train)-y_train.sum())} 条 ({(1-y_train.mean())*100:.2f}%)")
print(f"\n测试集：{X_test.shape[0]} 条")
print(f"  - 正样本：{int(y_test.sum())} 条 ({y_test.mean()*100:.2f}%)")
print(f"  - 负样本：{int(len(y_test)-y_test.sum())} 条 ({(1-y_test.mean())*100:.2f}%)")

# 5.2 标准化连续特征（用于逻辑回归、随机森林、XGBoost、LightGBM）
print("\n--- 连续特征标准化 ---")
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[continuous_features] = scaler.fit_transform(X_train[continuous_features])
X_test_scaled[continuous_features] = scaler.transform(X_test[continuous_features])
print(f"连续特征 {continuous_features} 已标准化（均值=0，标准差=1）")

# 5.3 准备CatBoost专用数据集（保留原始类别特征）
print("\n--- 准备CatBoost专用数据集 ---")
X_train_catboost = X_train.copy()
X_test_catboost = X_test.copy()

# 将类别特征转换为整数类型（CatBoost需要）
for col in categorical_features:
    X_train_catboost[col] = X_train_catboost[col].astype(int)
    X_test_catboost[col] = X_test_catboost[col].astype(int)

print(f"CatBoost数据集保留 {len(categorical_features)} 个原始类别特征")

# ==================== 6. 保存结果 ====================
print("\n" + "=" * 60)
print("【第六步】保存处理结果")
print("=" * 60)

# 6.1 保存清洗后的完整数据集
df_clean.to_csv(os.path.join(OUTPUT_DIR, "diabetes_cleaned.csv"), index=False)
print("✓ 已保存：diabetes_cleaned.csv（清洗后完整数据集）")

# 6.2 保存训练集和测试集（标准化版本，用于逻辑回归、随机森林、XGBoost、LightGBM）
pd.concat([X_train_scaled, y_train], axis=1).to_csv(
    os.path.join(OUTPUT_DIR, "train_scaled.csv"), index=False)
pd.concat([X_test_scaled, y_test], axis=1).to_csv(
    os.path.join(OUTPUT_DIR, "test_scaled.csv"), index=False)
print("✓ 已保存：train_scaled.csv / test_scaled.csv（标准化训练集/测试集）")

# 6.3 保存训练集和测试集（原始版本，用于CatBoost）
pd.concat([X_train_catboost, y_train], axis=1).to_csv(
    os.path.join(OUTPUT_DIR, "train_catboost.csv"), index=False)
pd.concat([X_test_catboost, y_test], axis=1).to_csv(
    os.path.join(OUTPUT_DIR, "test_catboost.csv"), index=False)
print("✓ 已保存：train_catboost.csv / test_catboost.csv（CatBoost训练集/测试集）")

# 6.4 保存标准化器
with open(os.path.join(OUTPUT_DIR, "scaler.pkl"), 'wb') as f:
    pickle.dump(scaler, f)
print("✓ 已保存：scaler.pkl（标准化器）")

# ==================== 7. 生成数据说明文档 ====================
print("\n" + "=" * 60)
print("【第七步】生成数据说明文档")
print("=" * 60)

doc_content = """# 糖尿病健康指标数据集说明文档

## 1. 数据来源
本数据集来源于美国疾病控制与预防中心（CDC）的**行为风险因素监测系统（BRFSS）2015年调查数据**。BRFSS是全球最大的持续性电话健康调查系统，每年收集超过40万条关于健康行为、慢性病和使用保健服务的数据。

## 2. 采集时间
数据采集时间为2015年。

## 3. 字段含义

| 字段名 | 描述 | 取值范围 | 数据类型 |
|--------|------|----------|----------|
| Diabetes_binary | 是否患有糖尿病（目标变量） | 0=否，1=是 | 整数 |
| HighBP | 是否患有高血压 | 0=否，1=是 | 整数 |
| HighChol | 是否患有高胆固醇 | 0=否，1=是 | 整数 |
| CholCheck | 过去5年内是否做过胆固醇检查 | 0=否，1=是 | 整数 |
| BMI | 身体质量指数 | 15-70（盖帽后） | 浮点数 |
| Smoker | 是否吸烟 | 0=否，1=是 | 整数 |
| Stroke | 是否曾被告知患有中风 | 0=否，1=是 | 整数 |
| HeartDiseaseorAttack | 是否患有冠心病或心肌梗死 | 0=否，1=是 | 整数 |
| PhysActivity | 过去30天内是否进行体育活动 | 0=否，1=是 | 整数 |
| Fruits | 是否每天食用水果 | 0=否，1=是 | 整数 |
| Veggies | 是否每天食用蔬菜 | 0=否，1=是 | 整数 |
| HvyAlcoholConsump | 是否重度饮酒 | 0=否，1=是 | 整数 |
| AnyHealthcare | 是否有任何形式的医疗保险 | 0=否，1=是 | 整数 |
| NoDocbcCost | 是否因费用问题无法看医生 | 0=否，1=是 | 整数 |
| GenHlth | 总体健康状况自评 | 1=优秀，2=非常好，3=好，4=一般，5=差 | 整数 |
| MentHlth | 过去30天内心理不健康的天数 | 0-30天 | 浮点数 |
| PhysHlth | 过去30天内身体不健康的天数 | 0-30天 | 浮点数 |
| DiffWalk | 是否有严重行走困难 | 0=否，1=是 | 整数 |
| Sex | 性别 | 0=女性，1=男性 | 整数 |
| Age | 年龄组 | 1=18-24岁，...，13=80岁以上 | 整数 |
| Education | 教育水平 | 1=从未上学，...，6=大学毕业 | 整数 |
| Income | 家庭年收入水平 | 1=<10000美元，...，8=>=75000美元 | 整数 |

## 4. 数据清洗规则

### 4.1 重复值处理
- 删除完全重复的行

### 4.2 缺失值处理
- 采用众数填充（若有缺失值）

### 4.3 异常值处理
- **BMI**：使用盖帽法，将小于1%分位数或大于99%分位数的值进行修正
- **MentHlth**：限制在0-30范围内
- **PhysHlth**：限制在0-30范围内

### 4.4 格式统一
- 二分类变量（HighBP、HighChol等）：转换为整数类型
- 序数分类变量（GenHlth、Age等）：转换为整数类型
- 连续变量（BMI、MentHlth、PhysHlth）：转换为浮点数类型

## 5. 特征工程说明

### 5.1 特征类型划分
- **类别特征**（18个）：HighBP、HighChol、CholCheck、Smoker、Stroke、HeartDiseaseorAttack、PhysActivity、Fruits、Veggies、HvyAlcoholConsump、AnyHealthcare、NoDocbcCost、GenHlth、DiffWalk、Sex、Age、Education、Income
- **连续特征**（3个）：BMI、MentHlth、PhysHlth

### 5.2 预处理方式
- **逻辑回归、随机森林、XGBoost、LightGBM**：连续特征标准化（均值=0，标准差=1），类别特征保留数值编码
- **CatBoost**：单独保留原始类别特征列，利用其原生类别特征处理能力

## 6. 数据集划分
- **划分比例**：训练集70%，测试集30%
- **抽样方式**：分层抽样（stratified），保证正负样本比例一致
- **随机种子**：random_state=42

## 7. 输出文件说明

| 文件名 | 描述 |
|--------|------|
| diabetes_cleaned.csv | 清洗后完整数据集 |
| train_scaled.csv | 标准化训练集（用于逻辑回归、随机森林、XGBoost、LightGBM） |
| test_scaled.csv | 标准化测试集（用于逻辑回归、随机森林、XGBoost、LightGBM） |
| train_catboost.csv | CatBoost训练集（保留原始类别特征） |
| test_catboost.csv | CatBoost测试集（保留原始类别特征） |
| scaler.pkl | 标准化器（用于反标准化） |

## 8. 合规性说明
- 本数据集来源于CDC公开发布的BRFSS调查数据，符合数据使用规范
- 数据已去除个人身份识别信息，仅保留匿名化的健康指标
- 数据处理过程遵循数据安全和隐私保护原则

---
*文档生成时间：2026年6月26日*
"""

# 保存文档
doc_path = os.path.join(OUTPUT_DIR, "数据说明文档.md")
with open(doc_path, 'w', encoding='utf-8') as f:
    f.write(doc_content)
print(f"✓ 已保存：{doc_path}")

# ==================== 完成 ====================
print("\n" + "=" * 60)
print("【全部完成】")
print("=" * 60)
print(f"\n所有输出文件已保存至：{OUTPUT_DIR}")
print("\n输出文件列表：")
print("  1. diabetes_cleaned.csv - 清洗后完整数据集")
print("  2. train_scaled.csv - 标准化训练集")
print("  3. test_scaled.csv - 标准化测试集")
print("  4. train_catboost.csv - CatBoost训练集")
print("  5. test_catboost.csv - CatBoost测试集")
print("  6. scaler.pkl - 标准化器")
print("  7. 数据说明文档.md - 数据说明文档")
