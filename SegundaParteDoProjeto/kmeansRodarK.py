import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import os

def main():
    # Path to the input file
    input_file = 'C:/MineracaoDados/adult/adultClear.data'
    output_file_zscore = 'C:/MineracaoDados/adult/DataNormalization_ZScore.data'
    output_file_kmeans = 'C:/MineracaoDados/adult/DataKMeans.csv'

    # Check if the input file exists
    if not os.path.exists(input_file):
        print(f"Input file {input_file} does not exist.")
        return

    # Column names for the DataFrame
    names = ['Age','Workclass','Education','Education-Num','Marital-Status',
             'Occupation','Relationship','Race','Sex','Capital-Gain',
             'Capital-Loss','Hours-per-week','Native-Country','Income']

    # Read the CSV file into a DataFrame
    df = pd.read_csv(input_file, names=names, usecols=['Age', 'Income'])
    
    # Show information about the original DataFrame
    ShowInformationDataFrame(df, "DataFrame Original")

    # Map the values in 'Income' column to 1 and 2
    df['Income'] = df['Income'].map({' <=50K': 1, ' >50K': 2})

    # Remove rows with non-numeric values in 'Income' column
    df = df.dropna()

    # Separate features for normalization
    x = df[['Age', 'Income']].values.astype(float)
    
    # Z-score normalization
    x_zscore = StandardScaler().fit_transform(x)

    # PCA
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x_zscore)
    principalDf = pd.DataFrame(data = principalComponents, columns = ['PC1', 'PC2'])

    # Variando o número de clusters (k) de 2 a 6
    for k in range(2, 7):
        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(x_zscore)
        
        # Adiciona os labels dos clusters ao DataFrame normalizado
        principalDf[f'Cluster_{k}'] = kmeans.labels_

    # Plot PCA para cada número de clusters
    PlotPCA(principalDf)

def PlotPCA(df):
    plt.figure(figsize=(15, 10))
    for k in range(2, 7):
        plt.subplot(2, 3, k-1)
        colors = ['blue', 'green', 'red', 'purple', 'orange']
        for cluster, color in zip(df[f'Cluster_{k}'].unique(), colors):
            plt.scatter(df[df[f'Cluster_{k}'] == cluster]['PC1'],
                        df[df[f'Cluster_{k}'] == cluster]['PC2'],
                        c=color, label=f'Cluster {cluster}')
        plt.title(f'PCA Plot with {k} KMeans Clusters')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend()

    plt.tight_layout()
    plt.show()

def ShowInformationDataFrame(df, message=""):
    print(message + "\n")
    print(df.info())
    print(df.describe())
    print(df.head(10))
    print("\n")

if __name__ == "__main__":
    main()
