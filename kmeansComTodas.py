import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
import os

def main():
    # Path to the input file
    input_file = 'C:\\MineraçãoDeDados\\adult\\adult - Copia.csv'
    output_file_zscore = 'C:\\MineraçãoDeDados\\adult\\DataNormalization_ZScore.data'
    output_file_minmax = 'C:\\MineraçãoDeDados\\adult\\DataNormalization_MinMax.data'
    output_file_kmeans = 'C:\\MineraçãoDeDados\\adult\\DataKMeans.csv'

    # Check if the input file exists
    if not os.path.exists(input_file):
        print(f"Input file {input_file} does not exist.")
        return

    # Column names for the DataFrame
    names = ['Age','Workclass','Education','Education-Num','Marital-Status',
             'Occupation','Relationship','Race','Sex','Capital-Gain',
             'Capital-Loss','Hours-per-week','Native-Country','Income']
    features = ['Age','Education-Num','Capital-Gain','Capital-Loss','Hours-per-week']
    target = 'Income'

    # Read the CSV file into a DataFrame
    df = pd.read_csv(input_file, names=names)
    
    # Show information about the original DataFrame
    ShowInformationDataFrame(df, "Dataframe original")

    # Separate features for normalization
    x = df.loc[:, features].values
    
    # Z-score normalization
    x_zscore = StandardScaler().fit_transform(x)
    normalized1Df = pd.DataFrame(data=x_zscore, columns=features)
    normalized1Df[target] = df[target]

    # Min-Max normalization
    x_minmax = MinMaxScaler().fit_transform(x)
    normalized2Df = pd.DataFrame(data=x_minmax, columns=features)
    normalized2Df[target] = df[target]

    # Save normalized DataFrames to CSV files
    try:
        normalized1Df.to_csv(output_file_zscore, index=False)
        print(f"Z-score normalized data saved to {output_file_zscore}")
        
        normalized2Df.to_csv(output_file_minmax, index=False)
        print(f"Min-Max normalized data saved to {output_file_minmax}")
    except Exception as e:
        print(f"An error occurred while saving the files: {e}")

    # --- Início da modificação para o algoritmo KMeans ---
    
    # Apply KMeans clustering
    n_clusters = 3  # Número de clusters desejados
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(x)
    
    # Adiciona os labels dos clusters ao DataFrame original
    df['Cluster'] = kmeans.labels_
    
    # Salva o DataFrame com os clusters em um arquivo CSV
    try:
        df.to_csv(output_file_kmeans, index=False)
        print(f"KMeans clustered data saved to {output_file_kmeans}")
    except Exception as e:
        print(f"An error occurred while saving the KMeans clustered file: {e}")
    
    # --- Fim da modificação para o algoritmo KMeans ---

def ShowInformationDataFrame(df, message=""):
    print(message + "\n")
    print(df.info())
    print(df.describe())
    print(df.head(10))
    print("\n")

if __name__ == "__main__":
    main()
