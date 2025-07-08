
import os
import pandas as pd
from model.pipeline.data_preparation import DataPrep
from model.pipeline.transformer import DataTransformer

def preprocess_data(raw_path='Real_Datasets/train_category_1.csv',
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
                    test_ratio=0.20,
                    save_path='./preprocess/processed.csv'):

    print("📂 Loading and processing raw dataset...")
    df = pd.read_csv(raw_path)

    # Preprocess raw DataFrame
    prep = DataPrep(df, categorical_columns, log_columns, mixed_columns, integer_columns, problem_type, test_ratio)
    transformed_df = prep.df

    # Transform to model input
    transformer = DataTransformer(train_data=transformed_df,
                                   categorical_list=prep.column_types['categorical'],
                                   mixed_dict=prep.column_types['mixed'])
    transformer.fit()
    transformed = transformer.transform(transformed_df.values)

    # Save processed data
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    pd.DataFrame(transformed).to_csv(save_path, index=False)
    print(f"✅ Saved processed data to {save_path}")
