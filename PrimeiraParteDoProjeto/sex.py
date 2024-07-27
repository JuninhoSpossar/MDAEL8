import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Carregar o arquivo CSV
data = pd.read_csv('C:\\MineraçãoDeDados\\adult\\adult.csv', delimiter=';')
print(data.columns.values.tolist())


# Remover espaços à esquerda dos valores na coluna 'Sex'
data['Sex'] = data['Sex'].str.lstrip()

# Criar variáveis dummy para a coluna "sex"
dummy_Sex = pd.get_dummies(data['Sex'], prefix='Sex')

# Concatenar as variáveis dummy ao dataframe original
data = pd.concat([data, dummy_Sex], axis=1)

# Remover a coluna original "sex"
data.drop('Sex', axis=1, inplace=True)

# Salvar o dataframe modificado em um novo arquivo CSV
data.to_csv('C:\\MineraçãoDeDados\\adult\\adultsexmodificado.csv', index=False)

# Mostrar a coluna nova
print(data['Sex_Male'])

# Imprimir as primeiras 10 linhas com as colunas
print(data.head(10))

