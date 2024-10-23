import pandas.api.types as ptypes

def min_max_scaling(column):
    return (column - column.min()) / (column.max() - column.min())


def preprocess_data(data_frame, target_columns, test_size_precent):
    for column in data_frame.columns:
        if ptypes.is_numeric_dtype(data_frame[column]):
            data_frame[column] = min_max_scaling(data_frame[column])
        else: 
            data_frame.get_dummies(data_frame, columns=[column])
    
    shuffled_data_frame = data_frame.sample(frac=1, random_state=42).reset_index(drop=True)
    test_size = int(len(data_frame) * test_size_precent)
    train_df = shuffled_data_frame.iloc[test_size:]
    test_df = shuffled_data_frame.iloc[:test_size]
    
    train_x = train_df.drop(columns=target_columns)
    train_y = train_df[target_columns]
    test_x = test_df.drop(columns=target_columns)
    test_y = test_df[target_columns]

    return train_x, train_y, test_x, test_y






