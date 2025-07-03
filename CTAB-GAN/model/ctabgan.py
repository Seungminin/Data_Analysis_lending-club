import pandas as pd
import time
import numpy as np
from scipy import stats
import tqdm
import wandb
import torch
from torch.utils.data import DataLoader, TensorDataset

# Used for pre/post-processing of the input/generated data
from model.pipeline.data_preparation import DataPrep 
# Model class for the CTABGANSynthesizer
from model.synthesizer.ctabgan_synthesizer import CTABGANSynthesizer 

import warnings
warnings.filterwarnings("ignore")


class CTABGAN():
    def __init__(self,
                 raw_csv_path="Real_Datasets/train_category_1.csv",
                 test_ratio=0.20,
                 categorical_columns=[
                     'debt_settlement_flag', 'sub_grade', 'home_ownership',
                     'purpose', 'grade', 'term_months'
                 ],
                 log_columns=[
                     'annual_inc', 'revol_bal', 'avg_cur_bal', 'installment'
                 ],
                 mixed_columns={
                     'last_fico_range_high': [0.0],
                     'mo_sin_old_rev_tl_op': [0.0],
                     'dti': [0.0],
                     'revol_util': [0.0]
                 },
                 integer_columns=[
                     'credit_history_years', 'term_months'
                 ],
                 problem_type={"Classification": 'loan_status'},
                 epochs=300):

        self.__name__ = 'CTABGAN'
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.synthesizer = CTABGANSynthesizer(epochs=epochs, device=self.device)
        self.raw_df = pd.read_csv(raw_csv_path)
        self.test_ratio = test_ratio
        self.categorical_columns = categorical_columns
        self.log_columns = log_columns
        self.mixed_columns = mixed_columns
        self.integer_columns = integer_columns
        self.problem_type = problem_type

    def fit(self):
        wandb.init(project="CTABGAN", name=f"ctabgan_{int(time.time())}", config={
        "epochs": self.synthesizer.epochs,
        "batch_size": self.synthesizer.batch_size,
        "random_dim": self.synthesizer.random_dim,
        "device": str(self.device)
    })
        start_time = time.time()
        self.data_prep = DataPrep(
            self.raw_df,
            self.categorical_columns,
            self.log_columns,
            self.mixed_columns,
            self.integer_columns,
            self.problem_type,
            self.test_ratio
        )

        self.synthesizer.fit(
            train_data=self.data_prep.df,
            categorical=self.data_prep.column_types["categorical"],
            mixed=self.data_prep.column_types["mixed"],
            type=self.problem_type
        )

        end_time = time.time()
        print('Finished training in', end_time - start_time, " seconds.")

    def generate_samples(self, N_CLS_PER_GEN=540000):
        syn_train = self.raw_df.copy()
        print(syn_train.shape)
        fraud_types = syn_train['loan_status'].unique()

        all_synthetic_data = pd.DataFrame()
        synthetic_subset = self.synthesizer.sample(num_samples=N_CLS_PER_GEN, fraud_types=fraud_types)
        synthetic_subset_df = pd.DataFrame(synthetic_subset, columns=self.data_prep.df.columns)

        all_synthetic_data = self.data_prep.inverse_prep(synthetic_subset_df)
        return all_synthetic_data
