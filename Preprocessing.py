import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_and_preprocess():
    df = pd.read_excel("DataSet.xls", header=1)
    df.rename(columns={df.columns[-1]: 'target'}, inplace=True)
    df.drop("ID", axis=1, inplace=True)

    X = df.drop('target', axis=1)
    y = df['target']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y.values
    







