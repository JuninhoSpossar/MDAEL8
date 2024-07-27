import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def ShowInformationDataFrame(df, message=""):
    print(message + "\n")
    print(df.info())
    print(df.describe())
    print(df.head(10))
    print("\n")

def main():
    # Faz a leitura do arquivo
    #input_file = 'C:\\MineraçãoDeDados\\Novapasta\\adult\\adult.csv'
    input_file = 'C:\\MineraçãoDeDados\\adult\\adultmodificacao4'
    names = ['Age', 'Workclass', 'fnlwgt', 'Education', 'Education-Num', 'Marital_Status', 'Occupation',
             'Relationship', 'Race', 'Sex_male', 'Capital_Gain', 'Capital_Loss', 'Hours_per_Week', 'Native_Country','Income']
    
    df = pd.read_csv(input_file, names=names)  # Nome das colunas
    ShowInformationDataFrame(df, "Dataframe original")

    # Separating out the features
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    x = df[numeric_features].values

    # Separating out the target
    target = 'Income'
    y = df[target].values

    # Standardizing the features
    x = StandardScaler().fit_transform(x)
    normalized_df = pd.DataFrame(data=x, columns=numeric_features)
    normalized_df[target] = df[target]  # Mantendo a coluna de destino

    # PCA projection
    pca = PCA(n_components=2)  # Reduzindo para 2 componentes para visualização
    principal_components = pca.fit_transform(x)
    principal_df = pd.DataFrame(data=principal_components, columns=['Principal Component 1', 'Principal Component 2'])
    final_df = pd.concat([principal_df, df[['Education-Num', target]]], axis=1)
    ShowInformationDataFrame(final_df, "Dataframe PCA")
   
    VisualizePcaProjection(final_df, target)


def VisualizePcaProjection(final_df, target_column):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('2 component PCA', fontsize=20)
    targets = final_df[target_column].unique()
    colors = ['r', 'g', 'b']
    labels = list(targets) + ['Education-Num']
    for target, color in zip(labels, colors):
        if target == 'Education-Num':
            ax.scatter(final_df['Education-Num'], final_df['Principal Component 1'], c=color, s=50)
        else:
            indicesToKeep = final_df[target_column] == target
            ax.scatter(final_df.loc[indicesToKeep, 'Principal Component 1'],
                       final_df.loc[indicesToKeep, 'Principal Component 2'],
                       c=color, s=50)
    ax.legend(labels)
    ax.grid()
    plt.show()
if __name__ == "__main__":
    main()