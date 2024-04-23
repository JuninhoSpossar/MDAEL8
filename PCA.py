import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def ShowInformationDataFrame(df, title):
    print(f"\n{title}:")
    print(df.head())
    print(df.dtypes)

def main():
    # Faz a leitura do arquivo
    input_file = 'C:\\MineraçãoDeDados\\adult\\adultmodificacao4'
    
    df = pd.read_csv(input_file)

    # Convert 'Sex_Male' and 'Education-num' to numeric
    df['Sex_Male'] = pd.to_numeric(df['Sex_Male'], errors='coerce')
    df['Education-num'] = pd.to_numeric(df['Education-num'], errors='coerce')


    ShowInformationDataFrame(df, "Dataframe original")

    # Separating out the features
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    x = df[numeric_features].values

    # Check the numeric columns
    print("Numeric columns:")
    print(df.select_dtypes(include=[np.number]).columns.tolist())

    # Separating out the target
    target = 'Income'
    if target in df.columns:
        y = df[target].values
    else:
        print(f"Target column '{target}' not found in dataframe.")

    # Standardizing the features
    if len(numeric_features) > 0:
        x = StandardScaler().fit_transform(x)
    else:
        print("No numeric features found to standardize.")

# PCA projection
    pca = PCA(n_components=2)  # Reduzindo para 2 componentes para visualização
    principal_components = pca.fit_transform(x)
    principal_df = pd.DataFrame(data=principal_components, columns=['Principal Component 1', 'Principal Component 2'])
    final_df = pd.concat([principal_df, df[target]], axis=1)
    ShowInformationDataFrame(final_df, "Dataframe PCA")
   
    VisualizePcaProjection(final_df, target)

    print("PCA components:")
    print(pca.components_)
    print("Explained variance ratio:")
    print(pca.explained_variance_ratio_)
    print("Numeric columns:")
    print(numeric_features)

def ShowInformationDataFrame(df, message=""):
    print(message + "\n")
    print(df.info())
    print(df.describe())
    print(df.head(10))
    print("\n")

def VisualizePcaProjection(final_df, target_column):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('2 component PCA', fontsize=20)
    targets = final_df[target_column].unique()
    colors = ['r', 'g']
    for target, color in zip(targets, colors):
        indicesToKeep = final_df[target_column] == target
        ax.scatter(final_df.loc[indicesToKeep, 'Principal Component 1'],
                   final_df.loc[indicesToKeep, 'Principal Component 2'],
                   c=color, s=50)
    ax.legend(targets)
    ax.grid()
    plt.show()


if __name__ == "__main__":
    main()