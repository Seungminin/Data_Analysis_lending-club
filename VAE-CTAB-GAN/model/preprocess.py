import os
import pandas as pd
from model.pipeline.data_preparation import DataPrep
from model.synthesizer.transformer import DataTransformer
import pickle

def preprocess_data(raw_path='Real_Datasets/train_category_1.csv',
        categorical_columns=[
            'purpose', 'home_ownership', 'loan_status', 'sub_grade',
            'grade', 'term_months', 'debt_settlement_flag'
        ],
        log_columns=['avg_cur_bal', 'installment', 'total_pymnt', 'total_pymnt_inv'],  #int_rate log 정규화를 통해 skew된 분포 완만하게 'int_rate'
        mixed_columns={  
            'annual_income': [0.0],
            'dti': [0.0],
            'revol_util': [0.0],
            'int_rate': [0.0],
            'loan_amnt' : [0.0],
            'funded_amnt' : [0.0]
            #'last_fico_range_high': [0.0]
        },

        single_gaussian_columns=['int_rate'],

        skew_multi_mode_columns=[
            'mo_sin_old_rev_tl_op','credit_history_years',
            'last_fico_range_high'
        ],

        integer_columns=['credit_history_years', 'term_months', 'last_fico_range_high'],
        problem_type={"Classification": 'loan_status'},
        test_ratio=0.20,
        save_path='./preprocess/processed_smotified.csv'):

    print(" Loading and processing raw dataset...")
    df = pd.read_csv(raw_path)
    mixed_columns_combined = mixed_columns.copy()
    for col in skew_multi_mode_columns + single_gaussian_columns:
        mixed_columns_combined[col] = [0.0]  # mode candidate
        
    prep = DataPrep(
        raw_df=df,
        categorical=categorical_columns,
        log=log_columns,
        mixed=mixed_columns,
        integer=integer_columns,
        type=problem_type,
        test_ratio=test_ratio,
        skew_columns=skew_multi_mode_columns,
        single_gaussian_columns=single_gaussian_columns
    )

    transformed_df = prep.df

    transformer = DataTransformer(train_data=transformed_df,
                                  categorical_list=prep.column_types['categorical'],
                                  mixed_dict=prep.column_types['mixed'],
                                  skewed_list=prep.column_types['skewed'],
                                  gaussian_list=prep.column_types['gaussian'])
    transformer.fit()
    transformed = transformer.transform(transformed_df.values)

    for item in transformer.output_info:
        print(item)
        
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    pd.DataFrame(transformed).to_csv(save_path, index=False)
    print(f" Saved processed data to {save_path}")

    os.makedirs('./preprocess/transformer', exist_ok=True)
    with open('./preprocess/transformer/transformer.pkl', 'wb') as f:
        pickle.dump(transformer, f)
    print(" Saved transformer to ./preprocess/transformer/transformer.pkl")

    os.makedirs('./preprocess/dataprep', exist_ok=True)
    with open('./preprocess/dataprep/dataprep.pkl', 'wb') as f:
        pickle.dump(prep, f)
    print("✅ Saved DataPrep object to ./preprocess/dataprep.pkl")