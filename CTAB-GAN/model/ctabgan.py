import pandas as pd
import time
import numpy as np
from scipy import stats
import tqdm

import torch
from torch.utils.data import DataLoader, TensorDataset

# Used for pre/post-processing of the input/generated data
from model.pipeline.data_preparation import DataPrep 
# Model class for the CTABGANSynthesizer
from model.synthesizer.ctabgan_synthesizer import CTABGANSynthesizer 

import warnings

warnings.filterwarnings("ignore")


class CTABGAN():

    """
    Generative model training class based on the CTABGANSynthesizer model

    Variables:
    1) raw_csv_path -> path to real dataset used for generation
    2) test_ratio -> parameter to choose ratio of size of test to train data
    3) categorical_columns -> list of column names with a categorical distribution
    4) log_columns -> list of column names with a skewed exponential distribution
    5) mixed_columns -> dictionary of column name and categorical modes used for "mix" of numeric and categorical distribution 
    6) integer_columns -> list of numeric column names without floating numbers  
    7) problem_type -> dictionary of type of ML problem (classification/regression) and target column name
    8) epochs -> number of training epochs

    Methods:
    1) __init__() -> handles instantiating of the object with specified input parameters
    2) fit() -> takes care of pre-processing and fits the CTABGANSynthesizer model to the input data 
    3) generate_samples() -> returns a generated and post-processed sythetic dataframe with the same size and format as per the input data 

    """
    
    def __init__(self,
                 raw_csv_path="Real_Datasets/Fraud_data.csv",
                 test_ratio=0.20,
                 categorical_columns=[
                     'Customer_Gender', 'Customer_personal_identifier', 'Customer_identification_number', 'Customer_credit_rating', 'Customer_loan_type', 
                     'Account_account_number', 'Account_account_type', 'Channel', 'Operating_System', 'Error_Code', 'Type_General_Automatic', 
                     'IP_Address', 'MAC_Address', 'Access_Medium', 'Location', 'Recipient_Account_Number', 'Fraud_Type'
                 ], 
                 log_columns=[],  # If there are no log-transformed columns, leave it as an empty list
                 mixed_columns={
                     'Account_one_month_std_dev': [0.0],
                     'Account_one_month_max_amount': [0.0], 
                     'Account_dawn_one_month_max_amount': [0.0], 
                     'Account_dawn_one_month_std_dev': [0.0]
                 },
                 integer_columns=[
                     'Customer_Birthyear', 'Customer_flag_change_of_authentication_1', 'Customer_flag_change_of_authentication_2',
                     'Customer_flag_change_of_authentication_3', 'Customer_flag_change_of_authentication_4', 
                     'Customer_rooting_jailbreak_indicator', 'Customer_mobile_roaming_indicator', 'Customer_VPN_Indicator', 
                     'Customer_flag_terminal_malicious_behavior_1', 'Customer_flag_terminal_malicious_behavior_2', 
                     'Customer_flag_terminal_malicious_behavior_3', 'Customer_flag_terminal_malicious_behavior_4', 
                     'Customer_flag_terminal_malicious_behavior_5', 'Customer_flag_terminal_malicious_behavior_6', 
                     'Customer_inquery_atm_limit', 'Customer_increase_atm_limit', 'Account_initial_balance', 'Account_balance', 
                     'Account_indicator_release_limit_excess', 'Account_amount_daily_limit', 'Account_indicator_Openbanking', 
                     'Account_remaining_amount_daily_limit_exceeded', 'Account_release_suspention', 'Transaction_Amount', 'Transaction_Failure_Status', 
                     'Transaction_num_connection_failure', 'Another_Person_Account', 'Unused_terminal_status', 
                     'Flag_deposit_more_than_tenMillion', 'Unused_account_status', 'Recipient_account_suspend_status', 
                     'Number_of_transaction_with_the_account', 'Transaction_history_with_the_account', 
                     'First_time_iOS_by_vulnerable_user', 'Customer_registration_datetime', 'Account_creation_datetime', 'Transaction_Datetime',
                     'Last_atm_transaction_datetime', 'Last_bank_branch_transaction_datetime', 'Transaction_resumed_date', 'Time_difference_seconds'
                 ],
                 problem_type={"Classification": 'Fraud_Type'},  # Adjust according to your classification target
                 epochs=1):

        self.__name__ = 'CTABGAN'
              
        self.synthesizer = CTABGANSynthesizer(epochs=epochs)
        self.raw_df = pd.read_csv(raw_csv_path)
        self.test_ratio = test_ratio
        self.categorical_columns = categorical_columns
        self.log_columns = log_columns
        self.mixed_columns = mixed_columns
        self.integer_columns = integer_columns
        self.problem_type = problem_type
        
    def fit(self):
          start_time = time.time()
          self.data_prep = DataPrep(self.raw_df,self.categorical_columns,self.log_columns,self.mixed_columns,self.integer_columns,self.problem_type,self.test_ratio)
          self.synthesizer.fit(train_data=self.data_prep.df, categorical = self.data_prep.column_types["categorical"], 
          mixed = self.data_prep.column_types["mixed"],type=self.problem_type)
          end_time = time.time()
          print('Finished training in',end_time-start_time," seconds.")
          
    def generate_samples(self, N_CLS_PER_GEN=3000):
        syn_train = self.raw_df.copy()
        print(syn_train.shape)
        fraud_types = syn_train['Fraud_Type'].unique()

        all_synthetic_data = pd.DataFrame()
        
        synthetic_subset = self.synthesizer.sample(num_samples=N_CLS_PER_GEN, fraud_types = fraud_types)
        synthetic_subset_df = pd.DataFrame(synthetic_subset, columns=self.data_prep.df.columns)
            
        all_synthetic_data = self.data_prep.inverse_prep(synthetic_subset_df)

        return all_synthetic_data


