# Importação das bibliotecas necessárias
import pandas as pd  # Manipulação de dados com DataFrames
import numpy as np  # Operações matemáticas e manipulação de arrays
from sklearn.preprocessing import StandardScaler, MinMaxScaler  # Normalização de dados
from sklearn.cluster import KMeans  # Algoritmo de clustering K-Means
from sklearn.model_selection import train_test_split  # Divisão dos dados em treino e teste
import os  # Operações com o sistema de arquivos
from sklearn.decomposition import PCA  # Decomposição em componentes principais (PCA)
import matplotlib.pyplot as plt  # Criação de gráficos

# Função para plotar os clusters em 2D após a projeção PCA
def plot_samples(projected, labels, title):
    fig = plt.figure()  # Cria uma nova figura para o gráfico
    u_labels = np.unique(labels)  # Obtém os rótulos únicos dos clusters
    for i in u_labels:  # Itera sobre cada rótulo
        # Cria um scatter plot para os dados correspondentes a cada cluster
        plt.scatter(projected[labels == i , 0], projected[labels == i , 1], 
                    label=i, edgecolor='none', alpha=0.5, cmap=plt.cm.get_cmap('tab10', 10))
    plt.xlabel('component 1')  # Rótulo do eixo X
    plt.ylabel('component 2')  # Rótulo do eixo Y
    plt.legend()  # Exibe a legenda
    plt.title(title)  # Define o título do gráfico

# Função principal onde todo o processamento é feito
def main():
    # Definição dos caminhos dos arquivos de entrada e saída
    input_file = './DataNumeric_AllColumns.data'  # Arquivo de entrada com os dados
    output_file_zscore = './DataNormalization_ZScore.data'  # Arquivo de saída para dados normalizados por Z-Score
    output_file_minmax = './DataNormalization_MinMax.data'  # Arquivo de saída para dados normalizados por Min-Max
    output_file_kmeans = './DataKMeans.csv'  # Arquivo de saída para dados clusterizados pelo K-Means

    # Verifica se o arquivo de entrada existe
    if not os.path.exists(input_file):
        print(f"Input file {input_file} does not exist.")  # Exibe uma mensagem de erro se o arquivo não for encontrado
        return  # Encerra a execução da função

    # Definição dos nomes das colunas do DataFrame
    names = ['Age', 'Workclass', 'Education', 'Education-Num', 'Marital-Status',
             'Occupation', 'Relationship', 'Race', 'Sex', 'Capital-Gain',
             'Capital-Loss', 'Hours-per-week', 'Native-Country', 'Income']
    
    # Seleção das colunas que serão usadas como características (features) para normalização
    features = ['Age', 'Education-Num', 'Capital-Gain', 'Capital-Loss', 'Hours-per-week']
    target = 'Income'  # Define a coluna alvo

    # Leitura do arquivo CSV em um DataFrame
    df = pd.read_csv(input_file, names=names, header=1)
    
    # Exibe informações sobre o DataFrame original
    ShowInformationDataFrame(df, "Dataframe original")

    # Separação das características (features) para normalização
    x = df.loc[:, features].values  # Obtém os valores das colunas selecionadas
    
    # Normalização Z-Score
    x_zscore = StandardScaler().fit_transform(x)  # Aplica a normalização Z-Score nos dados
    normalized1Df = pd.DataFrame(data=x_zscore, columns=features)  # Cria um novo DataFrame com os dados normalizados
    normalized1Df[target] = df[target]  # Adiciona a coluna alvo ao DataFrame

    # Normalização Min-Max
    x_minmax = MinMaxScaler().fit_transform(x)  # Aplica a normalização Min-Max nos dados
    normalized2Df = pd.DataFrame(data=x_minmax, columns=features)  # Cria um novo DataFrame com os dados normalizados
    normalized2Df[target] = df[target]  # Adiciona a coluna alvo ao DataFrame

    # Tentativa de salvar os DataFrames normalizados em arquivos CSV
    try:
        normalized1Df.to_csv(output_file_zscore, index=False)  # Salva o DataFrame normalizado por Z-Score
        print(f"Z-score normalized data saved to {output_file_zscore}")  # Mensagem de confirmação
        
        normalized2Df.to_csv(output_file_minmax, index=False)  # Salva o DataFrame normalizado por Min-Max
        print(f"Min-Max normalized data saved to {output_file_minmax}")  # Mensagem de confirmação
    except Exception as e:  # Captura qualquer exceção que ocorra
        print(f"An error occurred while saving the files: {e}")  # Exibe uma mensagem de erro

    # --- Início da modificação para o algoritmo KMeans ---
    
    # Aplica a decomposição PCA para reduzir os dados a 2 componentes principais
    pca = PCA(n_components=2)
    projected = pca.fit_transform(x_zscore)  # Aplica o PCA nos dados normalizados por Z-Score
    
    # Plota os dados originais projetados em 2D antes de aplicar o K-Means
    plot_samples(projected, df[target], 'Original data')
    plt.show()  # Exibe o gráfico

    # Divide os dados em conjuntos de treino e teste
    train_data, test_data = train_test_split(normalized1Df, test_size=0.2, random_state=0)
    
    # Define o número de clusters desejados para o K-Means
    n_clusters = 3
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)  # Instancia o K-Means
    kmeans.fit(projected)  # Ajusta o modelo K-Means aos dados projetados

    # Plota os dados após a aplicação do K-Means
    plot_samples(projected, kmeans.labels_, 'KMeans')
    plt.show()  # Exibe o gráfico
    
    # Tenta salvar o DataFrame original com os clusters em um arquivo CSV
    try:
        df.to_csv(output_file_kmeans, index=False)  # Salva o DataFrame no arquivo CSV
        print(f"KMeans clustered data saved to {output_file_kmeans}")  # Mensagem de confirmação
    except Exception as e:  # Captura qualquer exceção que ocorra
        print(f"An error occurred while saving the KMeans clustered file: {e}")  # Exibe uma mensagem de erro
    
    # --- Fim da modificação para o algoritmo KMeans ---

# Função para exibir informações sobre o DataFrame
def ShowInformationDataFrame(df, message=""):
    print(message + "\n")  # Exibe a mensagem passada como parâmetro
    print(df.info())  # Exibe informações gerais sobre o DataFrame
    print(df.describe())  # Exibe a descrição estatística do DataFrame
    print(df.head(10))  # Exibe as 10 primeiras linhas do DataFrame
    print("\n")

# Verifica se o script está sendo executado diretamente
if __name__ == "__main__":
    main()  # Chama a função principal para iniciar o processamento
