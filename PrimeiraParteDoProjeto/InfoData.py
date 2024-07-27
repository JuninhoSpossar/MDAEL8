import pandas as pd

# Ler o arquivo CSV
dataframe = pd.read_csv('C:\\MineraçãoDeDados\\adult\\adultmodificacao4')

# Imprimir informações sobre o dataframe
print("Informações sobre o dataframe:")
print("Número de linhas:", len(dataframe))
print("Número de colunas:", len(dataframe.columns))
print("Nomes das colunas:", dataframe.columns.tolist())
print("Tipos de dados das colunas:")
print(dataframe.dtypes)
print("Resumo estatístico do dataframe:")
print(dataframe.describe())