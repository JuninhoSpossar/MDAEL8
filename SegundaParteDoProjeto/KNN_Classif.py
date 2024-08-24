# Importação das bibliotecas necessárias
import pandas as pd  # Manipulação de dados com DataFrames
import numpy as np  # Operações matemáticas e manipulação de arrays
from sklearn.preprocessing import StandardScaler  # Normalização de dados
from sklearn.model_selection import train_test_split, cross_val_score  # Divisão dos dados e cross-validation
from sklearn.neighbors import KNeighborsClassifier  # Algoritmo K-NN
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score  # Métricas de avaliação
import os  # Operações com o sistema de arquivos
from sklearn.decomposition import PCA  # Decomposição em componentes principais (PCA)
import matplotlib.pyplot as plt  # Criação de gráficos
import itertools  # Biblioteca para manipulação de iteradores

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

# Função para exibir a matriz de confusão de forma gráfica
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Matriz de Confusão',
                          cmap=plt.cm.Blues):
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Matriz de confusão normalizada")
    else:
        print('Matriz de confusão, sem normalização')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Rótulo Verdadeiro')
    plt.xlabel('Rótulo Predito')

# Função principal onde todo o processamento é feito
def main():
    # Definição dos caminhos dos arquivos de entrada e saída
    input_file = './DataNumeric_AllColumns.data'  # Arquivo de entrada com os dados
    
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
    y = df[target].values  # Obtém os valores da coluna alvo
    
    # Normalização Z-Score
    x_zscore = StandardScaler().fit_transform(x)  # Aplica a normalização Z-Score nos dados

    # Divisão dos dados em conjuntos de treino (70%) e teste (30%)
    x_train, x_test, y_train, y_test = train_test_split(x_zscore, y, test_size=0.3, random_state=0)

    # Instanciação e treinamento do modelo K-NN
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(x_train, y_train)

    # Predição dos dados de teste
    y_pred = knn.predict(x_test)

    # Cálculo das métricas
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    print("Confusion Matrix:\n", cm)
    print("Accuracy:", accuracy)
    print("F1 Score:", f1)

    # Realizando Cross-Validation (k=10)
    cv_scores = cross_val_score(knn, x_zscore, y, cv=10)
    print("Cross-Validation Accuracy (k=10):", np.mean(cv_scores))

    # Plota a matriz de confusão
    plot_confusion_matrix(cm, classes=np.unique(y), title='Matriz de Confusão (Holdout)')

    # Aplica a decomposição PCA para reduzir os dados a 2 componentes principais
    pca = PCA(n_components=2)
    projected = pca.fit_transform(x_zscore)  # Aplica o PCA nos dados normalizados por Z-Score
    
    # Plota os dados originais projetados em 2D antes de aplicar o K-Means
    plot_samples(projected, df[target], 'Original data')
    plt.show()  # Exibe o gráfico

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
