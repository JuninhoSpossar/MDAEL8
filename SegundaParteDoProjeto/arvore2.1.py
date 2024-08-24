# Importação das bibliotecas necessárias
import pandas as pd  # Manipulação de dados com DataFrames
import numpy as np  # Operações matemáticas e manipulação de arrays
from sklearn.preprocessing import StandardScaler  # Normalização de dados
from sklearn.model_selection import train_test_split  # Divisão dos dados em treino e teste
from sklearn.tree import DecisionTreeClassifier, export_text  # Algoritmo de árvore de decisão
from sklearn.metrics import classification_report, confusion_matrix  # Métricas de avaliação
import os  # Operações com o sistema de arquivos
import matplotlib.pyplot as plt  # Criação de gráficos

# Função para exibir a matriz de confusão
def plot_confusion_matrix(cm, classes, title='Confusion Matrix'):
    plt.figure(figsize=(8, 6))  # Define o tamanho da figura
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)  # Plota a matriz de confusão
    plt.title(title)  # Define o título do gráfico
    plt.colorbar()  # Adiciona uma barra de cor para a escala
    tick_marks = np.arange(len(classes))  # Marcações dos ticks
    plt.xticks(tick_marks, classes, rotation=45)  # Define os rótulos dos ticks do eixo x
    plt.yticks(tick_marks, classes)  # Define os rótulos dos ticks do eixo y

    # Adiciona os textos dentro das células da matriz
    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in np.indices(cm.shape):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')  # Rótulo do eixo y
    plt.xlabel('Predicted label')  # Rótulo do eixo x
    plt.tight_layout()  # Ajusta o layout do gráfico

# Função principal onde todo o processamento é feito
def main():
    # Definição dos caminhos dos arquivos de entrada e saída
    input_file = './DataNumeric_AllColumns.data'  # Arquivo de entrada com os dados
    output_file_decision_tree = './DataDecisionTree.csv'  # Arquivo de saída para dados classificados pela árvore de decisão

    # Verifica se o arquivo de entrada existe
    if not os.path.exists(input_file):
        print(f"Input file {input_file} does not exist.")  # Exibe uma mensagem de erro se o arquivo não for encontrado
        return  # Encerra a execução da função

    # Definição dos nomes das colunas do DataFrame
    names = ['Age', 'Workclass', 'Education', 'Education-Num', 'Marital-Status',
             'Occupation', 'Relationship', 'Race', 'Sex', 'Capital-Gain',
             'Capital-Loss', 'Hours-per-week', 'Native-Country', 'Income']
    
    # Seleção das colunas que serão usadas como características (features)
    features = ['Age', 'Education-Num', 'Capital-Gain', 'Capital-Loss', 'Hours-per-week']
    target = 'Income'  # Define a coluna alvo

    # Leitura do arquivo CSV em um DataFrame
    df = pd.read_csv(input_file, names=names, header=1)
    
    # Exibe informações sobre o DataFrame original
    ShowInformationDataFrame(df, "Dataframe original")

    # Separação das características (features) e alvo (target)
    x = df.loc[:, features].values  # Obtém os valores das colunas selecionadas
    y = df[target].values  # Obtém os valores da coluna alvo
    
    # Normalização Z-Score
    x_zscore = StandardScaler().fit_transform(x)  # Aplica a normalização Z-Score nos dados

    # Divide os dados em conjuntos de treino e teste
    X_train, X_test, y_train, y_test = train_test_split(x_zscore, y, test_size=0.2, random_state=0)

    # Instancia e treina a árvore de decisão
    decision_tree = DecisionTreeClassifier(random_state=0)
    decision_tree.fit(X_train, y_train)

    # Faz predições no conjunto de teste
    y_pred = decision_tree.predict(X_test)

    # Avalia o modelo
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
    # Exibe a árvore de decisão em formato textual
    print("\nDecision Tree Structure:\n", export_text(decision_tree, feature_names=features))

    # Plota e exibe a matriz de confusão
    plot_confusion_matrix(cm, classes=['<=50K', '>50K'])
    plt.show()  # Exibe o gráfico
    
    # Salva as predições em um arquivo CSV
    try:
        results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
        results_df.to_csv(output_file_decision_tree, index=False)
        print(f"Decision Tree results saved to {output_file_decision_tree}")  # Mensagem de confirmação
    except Exception as e:  # Captura qualquer exceção que ocorra
        print(f"An error occurred while saving the Decision Tree results: {e}")  # Exibe uma mensagem de erro

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
