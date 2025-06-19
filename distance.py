import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

# 거리 행렬 계산
def distance_matrix(x):
    x = np.asarray(x).reshape(-1, 1)
    return np.abs(x - x.T)

# 이중 중심화
def double_center(matrix):
    row_mean = matrix.mean(axis=1, keepdims=True)
    col_mean = matrix.mean(axis=0, keepdims=True)
    total_mean = matrix.mean()
    return matrix - row_mean - col_mean + total_mean

# 거리 상관계수 계산
def distance_corr(x, y):
    a = double_center(distance_matrix(x))
    b = double_center(distance_matrix(y))
    dcov = np.sqrt(np.mean(a * b))
    dvar_x = np.sqrt(np.mean(a * a))
    dvar_y = np.sqrt(np.mean(b * b))
    return dcov / np.sqrt(dvar_x * dvar_y) if dvar_x > 0 and dvar_y > 0 else 0

# CSV 불러오기
df = pd.read_csv(r"C:\Users\조성찬\OneDrive - UOS\바탕 화면\철도연\DN210408-145917-CN1001-roadmapA.csv")
df_filtered = df.iloc[:, 2:].copy()

# 숫자 추출
def extract_numeric(s):
    if pd.isna(s):
        return None
    s = str(s).replace('．', '.').replace('－', '-')
    nums = re.findall(r"-?\d+(?:\.\d+)?", s)
    return float(nums[0]) if nums else None

df_numeric = df_filtered.apply(lambda col: col.map(extract_numeric))
df_no_nan = df_numeric.dropna(axis=1)
df_cleaned = df_no_nan.loc[:, df_no_nan.nunique() > 1]

# 거리 상관계수 계산
cols = df_cleaned.columns
dist_corr_matrix = pd.DataFrame(index=cols, columns=cols)

for i in cols:
    for j in cols:
        x, y = df_cleaned[i].dropna(), df_cleaned[j].dropna()
        length = min(len(x), len(y))
        dist_corr_matrix.loc[i, j] = distance_corr(np.array(x[:length]), np.array(y[:length]))

dist_corr_matrix = dist_corr_matrix.astype(float)

# ✅ 타겟 기준 변수
target_rows = ["Tc1 BC pressure", "Car2_BECU-BCP", "Car1_BECU-BCP"]
exclude_cols = set(target_rows)

# ✅ 각 타겟별 Top 10 관련 변수 추출 후 제외 목록 확장
for target in target_rows:
    if target not in dist_corr_matrix.index:
        continue
    top_corr = dist_corr_matrix.loc[target].drop(labels=target_rows, errors='ignore')
    top_10 = top_corr.sort_values(ascending=False).head(10).index
    filtered_top_10 = [col for col in top_10 if col not in target_rows]

# ✅ 히트맵 열에서 제외한 변수 제거
final_cols = [col for col in df_cleaned.columns if col not in exclude_cols]
filtered_corr = dist_corr_matrix.loc[target_rows, final_cols]

# ✅ 출력
for target in target_rows:
    if target not in dist_corr_matrix.index:
        continue
    print(f"\n📌 {target}와 거리 상관계수 높은 변수 Top 10:")
    top_corr = dist_corr_matrix.loc[target].drop(labels=target_rows, errors='ignore')
    for var, val in top_corr.sort_values(ascending=False).head(10).items():
        print(f"- {var:40s} : {val:.4f}")

# ✅ 히트맵 시각화
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(len(final_cols) * 0.8 + 5, 4))
sns.heatmap(filtered_corr, annot=True, fmt=".2f", cmap="coolwarm",
            xticklabels=True, yticklabels=True)
plt.title("Distance Correlation (타겟 및 관련 변수 제외)")
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# ✅ 제어의 근거 변수 목록
control_basis_vars = [
    "Speed",
    "Next Station code",
    "Distance",
    "ATO PWM",
    "Car2_BECU-ASP",
    "Car2_BECU-PWM",
    "Car1_BECU-PWM",
    "Car2_VVVF-Iq REF",
    "Car2_VVVF-BECU Brake Command"
]
control_basis_vars = [v for v in control_basis_vars if v in df_cleaned.columns]

# ✅ 제어 결과 vs 근거: 거리 상관계수 출력
for target in target_rows:
    if target not in dist_corr_matrix.index:
        continue
    print(f"\n📌 {target}과 제어의 근거 변수들과의 거리 상관계수 (내림차순):")
    top_corr = dist_corr_matrix.loc[target, control_basis_vars]
    for var in top_corr.sort_values(ascending=False).index:
        print(f"- {var:40s} : {top_corr[var]:.4f}")

# ✅ 제어 결과 vs 제어 근거 히트맵
subset_corr = dist_corr_matrix.loc[target_rows, control_basis_vars]
plt.figure(figsize=(len(control_basis_vars) * 0.8 + 5, len(target_rows) + 2))
sns.heatmap(subset_corr, annot=True, fmt=".2f", cmap="coolwarm",
            xticklabels=True, yticklabels=True)
plt.title("Distance Correlation (제어 결과 vs 제어 근거)")
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()