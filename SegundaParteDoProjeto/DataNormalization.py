import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def main():
    # Faz a leitura do arquivo
    input_file = 'C:/MineracaoDados/adult/adultClear.data'
    names = ['Age','Workclass','Education','Education-Num','Marital-Status',
             'Occupation','Relationship','Race','Sex','Capital-Gain',
             'Capital-Loss','Hours-per-week','Native-Country','Income']
    features = ['Age','Education-Num','Capital-Gain','Capital-Loss','Hours-per-week']
    target = 'Income'
    df = pd.read_csv(input_file, names=names)

    ShowInformationDataFrame(df, "Dataframe original")

    # Separando as features
    x = df.loc[:, features].values
    
    # Normalização Z-score
    x_zcore = StandardScaler().fit_transform(x)
    normalized1Df = pd.DataFrame(data=x_zcore, columns=features)
    normalized1Df[target] = df[target]

    # Normalização Min-Max
    x_minmax = MinMaxScaler().fit_transform(x)
    normalized2Df = pd.DataFrame(data=x_minmax, columns=features)
    normalized2Df[target] = df[target]

    # Salvar DataFrames normalizados em arquivos
    output_file_zscore = 'C:/MineracaoDados/adult/DataNormalization_ZScore.data'
    output_file_minmax = 'C:/MineracaoDados/adult/DataNormalization_MinMax.data'
    normalized1Df.to_csv(output_file_zscore, index=False)
    normalized2Df.to_csv(output_file_minmax, index=False)

def ShowInformationDataFrame(df, message=""):
    print(message+"\n")
    print(df.info())
    print(df.describe())
    print(df.head(10))
    print("\n")

if __name__ == "__main__":
    main()
