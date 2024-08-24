import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, homogeneity_score, completeness_score, v_measure_score
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
    k_values = range(2, 7)
    kmeans_results = []
    silhouette_scores = []
    homogeneity_scores = []
    completeness_scores = []
    v_measure_scores = []

    for k in k_values:
        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(x_zscore)
        kmeans_results.append(kmeans.labels_)

        # Calcula métricas de avaliação
        silhouette_avg = silhouette_score(x_zscore, kmeans.labels_)
        homogeneity = homogeneity_score(df['Income'], kmeans.labels_)
        completeness = completeness_score(df['Income'], kmeans.labels_)
        v_measure = v_measure_score(df['Income'], kmeans.labels_)

        silhouette_scores.append(silhouette_avg)
        homogeneity_scores.append(homogeneity)
        completeness_scores.append(completeness)
        v_measure_scores.append(v_measure)

    # Plot PCA com resultados de agrupamento para diferentes valores de k
    PlotPCA(principalDf, kmeans_results, k_values)

    # Mostrar métricas de avaliação
    print("Silhouette Scores:", silhouette_scores)
    print("Homogeneity Scores:", homogeneity_scores)
    print("Completeness Scores:", completeness_scores)
    print("V-measure Scores:", v_measure_scores)

def PlotPCA(df, kmeans_results, k_values):
    plt.figure(figsize=(15, 8))

    for i, k in enumerate(k_values):
        plt.subplot(2, 3, i+1)
        colors = ['blue', 'green', 'red', 'purple', 'orange', 'cyan']
        for cluster, color in zip(range(k), colors):
            plt.scatter(df['PC1'][kmeans_results[i] == cluster],
                        df['PC2'][kmeans_results[i] == cluster],
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
