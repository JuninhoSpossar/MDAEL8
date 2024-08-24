import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # Define o caminho do arquivo de entrada
    input_file = 'C:/MineracaoDados/adult/adultClear.data'
    
    # Verifica se o arquivo existe
    if not os.path.exists(input_file):
        print(f"Arquivo não encontrado: {input_file}")
        return
    
    # Define os nomes das colunas e as features para normalização
    nomes = ['Age', 'Workclass', 'Education', 'Education-Num', 'Marital-Status',
             'Occupation', 'Relationship', 'Race', 'Sex', 'Capital-Gain',
             'Capital-Loss', 'Hours-per-week', 'Native-Country', 'Income']
    features = ['Age', 'Education-Num', 'Capital-Gain', 'Capital-Loss', 'Hours-per-week']
    alvo = 'Income'
    
    # Lê o dataset
    df = pd.read_csv(input_file, names=nomes)
    
    # Mostra informações sobre o DataFrame original
    mostrar_informacoes_dataframe(df, "DataFrame Original")
    
    # Normaliza os dados usando Min-Max
    df_normalizado_minmax = normalizar_dados(df, features, alvo, metodo='minmax')
    
    # Define o caminho do arquivo de saída
    arquivo_saida_minmax = 'C:/MineracaoDados/adult/DataNormalization_MinMax.data'
    
    # Salva o DataFrame normalizado em arquivo
    df_normalizado_minmax.to_csv(arquivo_saida_minmax, index=False)
    
    # Mostra informações sobre o DataFrame normalizado
    mostrar_informacoes_dataframe(df_normalizado_minmax, "DataFrame Normalizado Min-Max")
    
    # Gera gráficos de medidas de tendência central
    gerar_graficos(df_normalizado_minmax, "Min-Max Normalizado")

def mostrar_informacoes_dataframe(df, mensagem=""):
    # Exibe informações sobre o DataFrame
    print(mensagem)
    print(df.info())
    print(df.describe())
    print(df.head(10))
    print("\n")

def normalizar_dados(df, features, alvo, metodo='minmax'):
    # Extrai os valores das features para normalização
    x = df.loc[:, features].values
    
    if metodo == 'zscore':
        scaler = StandardScaler()
    elif metodo == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError("Método de normalização desconhecido. Use 'zscore' ou 'minmax'.")
    
    x_normalizado = scaler.fit_transform(x)
    df_normalizado = pd.DataFrame(data=x_normalizado, columns=features)
    df_normalizado[alvo] = df[alvo]
    
    return df_normalizado

def gerar_graficos(df, titulo):
    # Configurações iniciais para os gráficos
    plt.figure(figsize=(15, 10))
    sns.set(style="whitegrid")

    # Plot para cada feature
    for i, feature in enumerate(df.columns[:-1], 1):
        plt.subplot(2, 3, i)
        sns.histplot(df[feature], kde=True, bins=30)
        plt.axvline(df[feature].mean(), color='r', linestyle='--', label='Média')
        plt.axvline(df[feature].median(), color='g', linestyle='-', label='Mediana')
        plt.legend()
        plt.title(f"{feature} - {titulo}")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
