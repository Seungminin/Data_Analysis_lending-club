from model.ctabgan import CTABGAN

if __name__ == "__main__":
    model = CTABGAN(
        raw_csv_path="Real_Datasets/train_category_1.csv",  # 🔁 데이터 경로
        epochs=300
    )
    
    model.fit()
    
    synthetic_df = model.generate_samples(N_CLS_PER_GEN=10)
    synthetic_df.to_csv("synthetic_output.csv", index=False)
