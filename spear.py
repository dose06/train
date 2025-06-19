import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import re

# 1. 파일 불러오기
df = pd.read_csv("DN210408-145917-CN1001-roadmapA.csv")
df_filtered = df.iloc[:, 2:].copy()

# 2. 숫자 추출 함수 정의
def extract_numeric(s):
    if pd.isna(s):
        return None
    s = str(s)
    s = s.replace('．', '.').replace('－', '-')
    nums = re.findall(r"-?\d+(?:\.\d+)?", s)
    return float(nums[0]) if nums else None

# 3. 숫자 변환 및 클렌징
df_numeric = df_filtered.applymap(extract_numeric)
df_no_nan = df_numeric.dropna(axis=1)
df_cleaned = df_no_nan.loc[:, df_no_nan.nunique() > 1]

# 4. 상관계수 계산
corr, _ = spearmanr(df_cleaned)
cols = df_cleaned.columns
corr_df = pd.DataFrame(corr, index=cols, columns=cols)

# 5. 기준 변수들
target_rows = ["Tc1 BC pressure", "Car2_BECU-BCP", "Car1_BECU-BCP"]





# 제외할 열: 타겟 3개만
exclude_cols = set(target_rows)

# 히트맵 열: 타겟 3개만 제외하고 전체 변수 사용
final_cols = [col for col in df_cleaned.columns if col not in exclude_cols]
filtered_corr = corr_df.loc[target_rows, final_cols]

# 📌 각 기준 변수별 상위 10개 상관관계 출력
# 📌 각 기준 변수별 상위 10개 상관관계 출력 (자기 자신뿐 아니라 다른 기준 변수도 제외)
for target in target_rows:
    print(f"\n📌 {target}과 상관계수 높은 변수 Top 10:")

    # target_rows 전체를 drop
    top_corr = corr_df.loc[target].drop(labels=target_rows, errors='ignore')
    top_corr_abs = top_corr.abs().sort_values(ascending=False)

    for var in top_corr_abs.head(10).index:
        value = corr_df.loc[target, var]
        print(f"- {var:40s} : {value:+.4f}")


# ✅ 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False  # 음수 기호 깨짐 방지

# 8. 히트맵 시각화
plt.figure(figsize=(len(final_cols) * 0.8 + 5, 4))
sns.heatmap(filtered_corr, annot=True, fmt=".2f", cmap="coolwarm",
            xticklabels=True, yticklabels=True)
plt.title("Spearman 상관관계 (타겟 및 관련 변수 제외)")
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
# 9. 제어의 근거 변수 정의
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

# 실제로 존재하는 변수만 필터링
control_basis_vars = [v for v in control_basis_vars if v in df_cleaned.columns]

# 10. 타겟 변수별 제어의 근거 변수와의 상관계수 출력
for target in target_rows:
    print(f"\n📌 {target}과 제어의 근거 변수들과의 Spearman 상관계수 (내림차순):")
    top_corr = corr_df.loc[target, control_basis_vars]
    top_corr_sorted = top_corr.abs().sort_values(ascending=False)

    for var in top_corr_sorted.index:
        value = corr_df.loc[target, var]
        print(f"- {var:40s} : {value:+.4f}")

# 11. 히트맵 (제어의 결과 vs 제어의 근거)
filtered_corr_subset = corr_df.loc[target_rows, control_basis_vars]

plt.figure(figsize=(len(control_basis_vars) * 0.8 + 5, len(target_rows) + 2))
sns.heatmap(filtered_corr_subset, annot=True, fmt=".2f", cmap="coolwarm",
            xticklabels=True, yticklabels=True)
plt.title("Spearman 상관관계 (제어의 결과 vs 제어의 근거)")
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
