import pandas as pd
import numpy as np


# Carrega o arquivo CSV
data = pd.read_csv('C:\\MineraçãoDeDados\\adult\\adultmodificacao4')

# Print the column names
print(data.columns)

# Imprime as 10 primeiras linhas e o nome das colunas
print(data.head(10))

# Deleta a coluna desejada
#data = data.drop('Education', axis=1)

# Salva o arquivo modificado
data.to_csv('C:\\MineraçãoDeDados\\adult\\adultmodificacao4', index=False)