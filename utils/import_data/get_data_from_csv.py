import pandas as pd

def get_data_from_csv(file_name):
    data_frame = pd.read_csv(f'samples/{file_name}')
    return data_frame