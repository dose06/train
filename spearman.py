import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import re
import sys
from scipy.stats import spearmanr
import datetime

# 📁 분석 폴더
folder = r"C:\Users\조성찬\OneDrive - UOS\바탕 화면\부산_1편성"

# 🔍 폴더 내 모든 .csv 파일
csv_files = [f for f in os.listdir(folder) if f.lower().endswith((".csv", ".xlsx"))][61:70]

# 🔧 통합 결과 저장용
all_results = []

# ✅ 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 숫자 추출 함수
def extract_numeric(s):
    if pd.isna(s):
        return None
    s = str(s).replace('．', '.').replace('－', '-').replace(',', '').strip()
    try:
        return float(s)
    except ValueError:
        nums = re.findall(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", s)
        return float(nums[0]) if nums else None

print("🔍 분석 대상 파일 목록:")
for f in csv_files:
    print("-", f)

# 🔁 파일별 분석
for file in csv_files:
    print(f"\n=== 📁 {file} 분석중 ===")
    file_path = os.path.join(folder, file)
    
    # 파일 불러오기
    if file.lower().endswith(".csv"):
        df = pd.read_csv(file_path)
    elif file.lower().endswith(".xlsx"):
        df = pd.read_excel(file_path)
    else:
        continue

    df_filtered = df.iloc[:, 2:].copy()
    df_numeric = df_filtered.applymap(extract_numeric)
    df_numeric = df_numeric.dropna(axis=1)
    df_numeric = df_numeric.loc[:, df_numeric.nunique() > 1]

    # 타겟 변수
    target_rows = [
        "Tc1 BC pressure",
        "Car2_BECU-BCP",
        "Car1_BECU-BCP",
        "Car3_BECU-BCP",
        "Car4_BECU-BCP",
        "Car5_BECU-BCP",
        "Car6_BECU-BCP",
        "Car7_BECU-BCP",
        "Car8_BECU-BCP"
    ]

    print("✅ 최종 df_numeric 컬럼:", df_numeric.columns.tolist())
    print("✅ 타겟 컬럼별 NaN 비율:")
    for target in target_rows:
        if target in df_numeric.columns:
            nan_ratio = df_numeric[target].isna().mean()
            print(f"🔎 {target} NaN 비율: {nan_ratio:.2%}")
        else:
            print(f"{target}: 컬럼 없음")

    # 🔎 타겟별 상관계수 계산 (비타겟 변수만)
    for target in target_rows:
        if target not in df_numeric.columns:
            print(f"⚠️ {target} not found in {file}")
            continue

        target_series = df_numeric[target]

        for col in df_numeric.columns:
            if col == target or col in target_rows:
                continue

            col_series = df_numeric[col]
            common = target_series.notna() & col_series.notna()
            if common.sum() < 2:
                continue

            corr, _ = spearmanr(target_series[common], col_series[common])

            # ✅ 각 타겟-변수 쌍별로 저장
            all_results.append({
                "file": file,
                "target": target,
                "variable": col,
                "corr": corr
            })

# ✅ DataFrame으로 통합
results_df = pd.DataFrame(all_results)
print("\n=== results_df preview ===")
print(results_df.head())
print("Columns:", results_df.columns.tolist())
print("Empty:", results_df.empty)

# ✅ 빈 경우 종료
if results_df.empty:
    print("❌ 결과가 없습니다. 타겟 변수명을 확인해주세요.")
    sys.exit()

# 🔬 변수별 전체 평균 상관계수 계산
print("\n=== 변수별 전체 파일 평균 상관계수 ===")
grouped = results_df.groupby(['target','variable'])['corr'].mean().reset_index()
grouped_sorted = grouped.sort_values(
    by=['target','corr'],
    ascending=[True, False],
    key=lambda col: col if col.name != 'corr' else col.abs()
)

for target in target_rows:
    target_df = grouped_sorted[grouped_sorted['target'] == target]
    print(f"\n▶️ {target} 상위 변수 (평균 상관계수 순):")
    print(target_df.head(5))

# ✅ 결과 저장
now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = os.path.join(folder, f"부산_1편성_변수별평균_상관분석결과_{now}.csv")

grouped_sorted.to_csv(output_path, index=False)
print(f"\n✅ 통합 분석 완료. 결과를 '{output_path}'로 저장했습니다.")
