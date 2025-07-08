from model.ctabgan import CTABGAN

if __name__ == "__main__":
    model = CTABGAN(
        raw_csv_path="Real_Datasets/train_category_1.csv",
        epochs=300
    )
    
    model.fit()
    
    synthetic_df = model.generate_samples(N_CLS_PER_GEN=540000)
    output_path = "C:/Users/GCU/Lending_club/Data_Analysis_lending-club/CTAB-GAN/Fake_Datasets/"
    synthetic_df.to_csv(path_or_buf=output_path+"ctab-gan_300epochs.csv", index=False)
