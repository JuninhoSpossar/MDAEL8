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

    # Salvar uma cópia do DataFrame original
    df_original = df.copy()

    # Converter colunas categóricas em numéricas
    cat_columns = df.select_dtypes(['object']).columns
    df[cat_columns] = df[cat_columns].apply(lambda x: pd.factorize(x)[0])

    # Exibir informações antes e depois da modificação
    ShowInformationDataFrame(df_original, df, "Comparação entre DataFrame original e modificado")

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

    # Salvar DataFrame modificado com todas as colunas numéricas
    output_file_numeric = 'C:/MineracaoDados/adult/DataNumeric_AllColumns.data'
    df.to_csv(output_file_numeric, index=False)

def ShowInformationDataFrame(df_original, df_modified, message=""):
    print(message+"\n")
    
    print("DataFrame Original:")
    print(df_original.head(10))
    print("\n")
    
    print("DataFrame Modificado:")
    print(df_modified.head(10))
    print("\n")

if __name__ == "__main__":
    main()
