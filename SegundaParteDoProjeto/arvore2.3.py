# Importação das bibliotecas necessárias
import pandas as pd  # Manipulação de dados com DataFrames
import numpy as np  # Operações matemáticas e manipulação de arrays
from sklearn.preprocessing import StandardScaler  # Normalização de dados
from sklearn.model_selection import train_test_split  # Divisão dos dados em treino e teste
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text  # Algoritmo de árvore de decisão
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score  # Métricas de avaliação
import os  # Operações com o sistema de arquivos
import matplotlib.pyplot as plt  # Criação de gráficos
import itertools  # Ferramentas para iteração

# Função para exibir a matriz de confusão
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
    plt.show()

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

    # Instancia e treina a árvore de decisão com parâmetros mais restritivos
    decision_tree = DecisionTreeClassifier(random_state=0, max_depth=3, min_samples_leaf=20, min_samples_split=20)
    decision_tree.fit(X_train, y_train)

    # Faz predições no conjunto de teste
    y_pred = decision_tree.predict(X_test)

    # Avalia o modelo
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Calcula e exibe a Acurácia e F1 Score
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')  # Calcula o F1 Score ponderado
    print(f"Acurácia: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Exibe a árvore de decisão em formato textual
    print("\nDecision Tree Structure:\n", export_text(decision_tree, feature_names=features))

    # Plota a árvore de decisão
    plt.figure(figsize=(20,10))  # Define o tamanho da figura
    plot_tree(decision_tree, feature_names=features, class_names=['<=50K', '>50K'], filled=True)
    plt.title('Decision Tree Visualization')  # Define o título do gráfico
    plt.show()  # Exibe o gráfico
    
    # Plota a matriz de confusão
    plot_confusion_matrix(cm, classes=['<=50K', '>50K'])

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
