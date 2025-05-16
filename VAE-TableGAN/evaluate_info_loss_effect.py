import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
import matplotlib.pyplot as plt
import os

def matrix_distance_euclidian(ma, mb):
    return np.sqrt(np.sum(np.power(np.subtract(ma, mb), 2)))

def compute_mmd(real, fake, gamma=1.0):
    """Compute MMD with RBF kernel"""
    xx = rbf_kernel(real, real, gamma=gamma)
    yy = rbf_kernel(fake, fake, gamma=gamma)
    xy = rbf_kernel(real, fake, gamma=gamma)
    return xx.mean() + yy.mean() - 2 * xy.mean()

# 실험 설정
experiment_configs = [
    {"id": "test1", "delta": 0.0},
    {"id": "test2", "delta": 0.1},
    {"id": "test3", "delta": 0.2},
]

results = []

# 분석 루프
for config in experiment_configs:
    test_id = config["id"]
    fake_path = f"./samples/{test_id}_generated.csv"
    real_path = "./data/loan/loan.csv"

    if not os.path.exists(fake_path) or not os.path.exists(real_path):
        print(f"파일 없음: {test_id}")
        continue

    real_df = pd.read_csv(real_path)
    fake_df = pd.read_csv(fake_path)

    # 숫자형 공통 컬럼만 정제
    shared_cols = list(set(real_df.columns) & set(fake_df.columns))
    real_df = real_df[shared_cols].select_dtypes(include=[np.number]).dropna()
    fake_df = fake_df[shared_cols].select_dtypes(include=[np.number]).dropna()

    min_len = min(len(real_df), len(fake_df))
    real_df = real_df.iloc[:min_len]
    fake_df = fake_df.iloc[:min_len]

    # Corr 차이
    corr_diff = np.mean(np.abs(real_df.corr().values - fake_df.corr().values))

    # MMD 계산
    mmd_score = compute_mmd(real_df.values, fake_df.values)

    results.append({
        "test_id": test_id,
        "delta_m": config["delta"],
        "corr_diff": corr_diff,
        "mmd_score": mmd_score
    })

# 결과 테이블 및 시각화
results_df = pd.DataFrame(results)
print("\n📊 Info Loss 영향 비교:\n", results_df)

plt.figure(figsize=(8, 5))
plt.plot(results_df["delta_m"], results_df["corr_diff"], marker='o', label="Correlation Difference")
plt.plot(results_df["delta_m"], results_df["mmd_score"], marker='x', label="MMD Score")
plt.xlabel("Delta Mean/Var (정규화 임계값)")
plt.ylabel("Score")
plt.title("Info Loss vs. Correlation Diff / MMD")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("info_loss_analysis.png")
plt.show()
