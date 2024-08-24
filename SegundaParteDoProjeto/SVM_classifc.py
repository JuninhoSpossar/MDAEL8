# Importação das bibliotecas necessárias
import pandas as pd  # Manipulação de dados com DataFrames
import numpy as np  # Operações matemáticas e manipulação de arrays
from sklearn.preprocessing import StandardScaler  # Normalização de dados
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict  # Divisão dos dados e validação cruzada
from sklearn.svm import SVC  # Algoritmo SVM
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score  # Métricas de avaliação
import os  # Operações com o sistema de arquivos
import matplotlib.pyplot as plt  # Criação de gráficos
import itertools  # Manipulação de iteradores

# Função para exibir informações sobre o DataFrame
def ShowInformationDataFrame(df, message=""):
    print(message + "\n")  # Exibe a mensagem passada como parâmetro
    print(df.info())  # Exibe informações gerais sobre o DataFrame
    print(df.describe())  # Exibe a descrição estatística do DataFrame
    print(df.head(10))  # Exibe as 10 primeiras linhas do DataFrame
    print("\n")

# Função para plotar a matriz de confusão
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
    plt.show()  # Exibe o gráfico

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

    # Separação das características (features) e do alvo (target)
    X = df.loc[:, features].values  # Obtém os valores das colunas selecionadas
    y = df[target].values  # Obtém os valores da coluna alvo
    
    # Normalização Z-Score
    scaler = StandardScaler()
    X_zscore = scaler.fit_transform(X)  # Aplica a normalização Z-Score nos dados

    # Divisão dos dados em conjuntos de treino e teste (Holdout)
    X_train, X_test, y_train, y_test = train_test_split(X_zscore, y, test_size=0.3, random_state=0)

    # Instancia o modelo SVM
    svm_model = SVC(kernel='linear', random_state=0)

    # Treina o modelo usando o conjunto de treino
    svm_model.fit(X_train, y_train)

    # Faz as predições no conjunto de teste
    y_pred = svm_model.predict(X_test)

    # Calcula e exibe as métricas no conjunto de teste (Holdout)
    print("Matriz de Confusão (Holdout):")
    cm_holdout = confusion_matrix(y_test, y_pred)
    print(cm_holdout)
    print(f"Acurácia (Holdout): {accuracy_score(y_test, y_pred):.4f}")
    print(f"F1 Score (Holdout): {f1_score(y_test, y_pred, average='weighted'):.4f}")

    # Plota a matriz de confusão para o conjunto de teste (Holdout)
    plot_confusion_matrix(cm_holdout, classes=np.unique(y), title='Matriz de Confusão (Holdout)')

    # Cross-Validation com k=10
    y_pred_cv = cross_val_predict(svm_model, X_zscore, y, cv=10)

    # Calcula e exibe as métricas no Cross-Validation
    print("\nMatriz de Confusão (Cross-Validation):")
    cm_cv = confusion_matrix(y, y_pred_cv)
    print(cm_cv)
    print(f"Acurácia (Cross-Validation): {accuracy_score(y, y_pred_cv):.4f}")
    print(f"F1 Score (Cross-Validation): {f1_score(y, y_pred_cv, average='weighted'):.4f}")

    # Plota a matriz de confusão para a validação cruzada (Cross-Validation)
    plot_confusion_matrix(cm_cv, classes=np.unique(y), title='Matriz de Confusão (Cross-Validation)')

# Verifica se o script está sendo executado diretamente
if __name__ == "__main__":
    main()  # Chama a função principal para iniciar o processamento
