import pandas as pd
import time
import torch
import wandb
from model.pipeline.data_preparation import DataPrep
from model.synthesizer.ctabgan_synthesizer import CTABGANSynthesizer


class CTABGAN():
    def __init__(self,
                 raw_csv_path="Real_Datasets/train_category_1.csv",
                 test_ratio=0.20,
                 categorical_columns=[],
                 log_columns=[],
                 mixed_columns={},
                 integer_columns=[],
                 problem_type=None,
                 epochs=300,
                 batch_size=100):

        self.__name__ = 'CTABGAN'
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.synthesizer = CTABGANSynthesizer(epochs=epochs, batch_size=batch_size, device=self.device)
        self.raw_df = pd.read_csv(raw_csv_path)
        self.test_ratio = test_ratio
        self.categorical_columns = categorical_columns
        self.log_columns = log_columns
        self.mixed_columns = mixed_columns
        self.integer_columns = integer_columns
        self.problem_type = problem_type

    def fit(self, df, transformer, condvec, sampler, output_info, target_index=None):
        start_time = time.time()
        self.synthesizer.fit(
            train_data=df,
            transformer=transformer,
            condvec=condvec,
            sampler=sampler,
            output_info=output_info,
            problem_type=self.problem_type,
            target_index=target_index
        )
        end_time = time.time()
        print(' Finished training in', end_time - start_time, "seconds.")

    def generate_samples(self, N_CLS_PER_GEN=540000):
        syn_train = self.raw_df.copy()
        fraud_types = syn_train['loan_status'].unique()
        synthetic_subset = self.synthesizer.sample(num_samples=N_CLS_PER_GEN, fraud_types=fraud_types)
        synthetic_subset_df = pd.DataFrame(synthetic_subset, columns=self.data_prep.df.columns)
        all_synthetic_data = self.data_prep.inverse_prep(synthetic_subset_df)
        return all_synthetic_data
