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

    # Apply KMeans clustering
    n_clusters = 3  # Número de clusters desejados
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(x_zscore)
    
    # Adiciona os labels dos clusters ao DataFrame normalizado
    principalDf['Cluster'] = kmeans.labels_
    
    # Plot PCA
    PlotPCA(principalDf, df)

def PlotPCA(df, original_df):
    plt.figure(figsize=(10, 6))
    colors = ['blue', 'green', 'red']
    income_colors = {1: 'yellow', 2: 'orange'}
    for cluster, color in zip(df['Cluster'].unique(), colors):
        plt.scatter(df[df['Cluster'] == cluster]['PC1'],
                    df[df['Cluster'] == cluster]['PC2'],
                    c=color, label=f'Cluster {cluster}')
    for income, color in income_colors.items():
        plt.scatter(original_df[original_df['Income'] == income]['Age'],
                    original_df[original_df['Income'] == income]['Income'],
                    c=color, label=f'Income {income}', alpha=0.5)
    plt.title('PCA Plot with KMeans Clusters and Income')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.show()

def ShowInformationDataFrame(df, message=""):
    print(message + "\n")
    print(df.info())
    print(df.describe())
    print(df.head(10))
    print("\n")

if __name__ == "__main__":
    main()
