import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

# ----------------------------
# 1. 输入你的数据
# ----------------------------
data = {
    "train_loss": [
        4.4057, 4.4893, 4.5652, 4.5961, 4.6522, 4.6585, 4.67, 4.7088, 4.7188,
        4.7379, 4.7604, 4.7714, 4.7731, 4.7732, 4.7746, 4.7819, 4.7889, 4.789,
        4.791, 4.8013, 4.8049, 4.8178, 4.8427, 4.843
    ],
    "eval_loss": [
        4.3404, 4.40985, 4.50623, 4.52609, 4.5812, 4.59967, 4.61579, 4.65024,
        4.65827, 4.67717, 4.69649, 4.71305, 4.70868, 4.72437, 4.70937, 4.73263,
        4.72776, 4.72535, 4.72746, 4.74891, 4.74348, 4.75909, 4.78707, 4.78249
    ],
    "learning_rate": [
        0.00089281, 0.00052191, 0.0032674, 0.00049316, 0.0040781, 0.0049001,
        0.0018627, 0.0023581, 0.0029283, 0.0023171, 0.0021772, 0.0049646,
        0.0042777, 0.004374, 0.0049703, 0.0036987, 0.0041517, 0.0047454,
        0.0043112, 0.0031593, 0.0029181, 0.0041666, 0.0049725, 0.0035566
    ],
    "classifier_amplifier": [
        1303.49874, 254.78324, 167.8778, 1142.24055, 740.73515, 275.58329,
        369.05634, 1751.89734, 463.03568, 1073.59168, 354.83661, 158.53793,
        235.4077, 670.12743, 710.56036, 1387.25023, 189.15747, 110.92016,
        155.33434, 447.71149, 387.53764, 124.8375, 147.73211, 376.79764
    ],
    "binary_factor_scaling": [
        0.46401, 0.11669, 3.29894, 0.34172, 0.37418, 0.50349, 0.28511, 0.31667,
        1.81093, 0.2216, 0.49454, 0.11489, 0.78908, 2.32973, 0.26985, 3.0344,
        0.93109, 0.36039, 1.164, 0.24341, 0.46124, 1.12658, 5.00557, 0.40505
    ],
    "ternary_factor_scaling": [
        0.14968, 0.16464, 0.081058, 0.15352, 0.056664, 0.062273, 0.062095,
        0.063339, 0.043018, 0.045426, 0.055887, 0.040939, 0.034482, 0.041347,
        0.037473, 0.037096, 0.036991, 0.030829, 0.043271, 0.04003, 0.041002,
        0.033505, 0.047639, 0.0356
    ]
}

df = pd.DataFrame(data)

# ----------------------------
# 2. 可视化：ternary & lr vs eval_loss
# ----------------------------
plt.figure(figsize=(12, 5))

# Ternary vs Loss
plt.subplot(1, 2, 1)
sns.scatterplot(data=df, x='ternary_factor_scaling', y='eval_loss', hue='eval_loss', palette='viridis')
plt.title('Ternary Scaling vs Eval Loss')
plt.xlabel('Ternary Factor Scaling')
plt.ylabel('Eval Loss')

# LR vs Loss
plt.subplot(1, 2, 2)
sns.scatterplot(data=df, x='learning_rate', y='eval_loss', hue='eval_loss', palette='viridis')
plt.title('Learning Rate vs Eval Loss')
plt.xlabel('Learning Rate')
plt.ylabel('Eval Loss')
plt.xscale('log')
plt.tight_layout()
plt.show()

# ----------------------------
# 3. 拟合 lr ~ ternary^(-alpha) 关系
# ----------------------------
# 取 top 10 runs (lowest eval_loss)
top_k = 10
df_top = df.nsmallest(top_k, 'eval_loss')

x = np.log(df_top['ternary_factor_scaling'])
y = np.log(df_top['learning_rate'])

slope, intercept, r_value, p_value, std_err = linregress(x, y)
alpha = -slope
C = np.exp(intercept)

print(f"\n📊 拟合关系: learning_rate ≈ C / (ternary^α)")
print(f"   α = {alpha:.3f}")
print(f"   C = {C:.5f}")
print(f"   R² = {r_value**2:.3f} (p={p_value:.3f})")

# 可视化拟合曲线
ternary_range = np.linspace(0.03, 0.18, 50)
lr_pred = C / (ternary_range ** alpha)

plt.figure(figsize=(6, 4))
plt.scatter(df['ternary_factor_scaling'], df['learning_rate'], c=df['eval_loss'], cmap='coolwarm')
plt.plot(ternary_range, lr_pred, 'r--', label=f'Fitted: lr ∝ ternary^(-{alpha:.2f})')
plt.colorbar(label='Eval Loss')
plt.xlabel('Ternary Factor Scaling')
plt.ylabel('Learning Rate')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.title('Compensation Effect: LR vs Ternary')
plt.show()

# ----------------------------
# 4. 特征重要性验证（用 Random Forest）
# ----------------------------
X = df[['learning_rate', 'ternary_factor_scaling', 'classifier_amplifier', 'binary_factor_scaling']]
y = df['eval_loss']

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X, y)

importances = rf.feature_importances_
feature_names = X.columns

# Plot
plt.figure(figsize=(6, 4))
indices = np.argsort(importances)[::-1]
plt.bar(range(len(importances)), importances[indices])
plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
plt.title('Feature Importance (Random Forest)')
plt.tight_layout()
plt.show()

print("\n🔍 Random Forest Feature Importance:")
for i in indices:
    print(f"  {feature_names[i]}: {importances[i]:.3f}")

# ----------------------------
# 5. 推荐下一组 promising 超参
# ----------------------------
print("\n🎯 推荐下一阶段搜索范围（基于拟合）:")
print(f"  ternary_factor_scaling ∈ [0.05, 0.18]")
print(f"  learning_rate ≈ {C:.5f} / (ternary^({alpha:.2f}))")
print(f"     → 当 ternary=0.10, lr ≈ {C / (0.10**alpha):.5f}")
print(f"     → 当 ternary=0.15, lr ≈ {C / (0.15**alpha):.5f}")

# 固定其他参数
print("\n🔧 建议固定参数:")
print("  binary_factor_scaling = 0.35")
print("  classifier_amplifier = 1000")

# 6.可视化
# ----------------------------

plt.figure(figsize=(6,4))
plt.scatter(df['ternary_factor_scaling'], df['learning_rate'], 
            c=df['eval_loss'], cmap='coolwarm', label='Runs')
ternary_fine = np.linspace(0.03, 0.18, 100)
lr_fit = 0.00006 / (ternary_fine ** 1.374)
plt.plot(ternary_fine, lr_fit, 'k--', label='Fitted curve')
plt.xscale('log'); plt.yscale('log')
plt.xlabel('Ternary Scaling'); plt.ylabel('Learning Rate')
plt.colorbar(label='Eval Loss')
plt.legend()
plt.title('Strong Compensation Effect (R²=0.72)')
plt.show()

# 保存图片
plt.savefig('compensation_effect.png', dpi=300)