import pandas as pd

# Ler o arquivo
data = pd.read_csv('C:\\MineraçãoDeDados\\adult\\adultmodificacao2.csv')

# Verificar o tipo de dados da coluna 'Sex_Male'
print(data['Sex_Male'].dtype)

# Converter a coluna 'Sex_Male' para int
data['Sex_Male'] = data['Sex_Male'].astype(int)

#Verificar o tipo de dados da coluna 'Sex_Male' novamente
print(data['Sex_Male'].dtype)

# Imprimir as 10 primeiras linhas da coluna 'Sex_Male'
print(data['Sex_Male'].head(10))

# Salvar o dataframe modificado em um novo arquivo CSV
data.to_csv('C:\\MineraçãoDeDados\\adult\\adultmodificacao3.csv', index=False)
