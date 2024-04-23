import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Suponha que 'dados' é o seu DataFrame e 'Workclass' é a coluna com dados categóricos
dados = pd.read_csv('C:\\MineraçãoDeDados\\adult\\adultmodificacao4')

# Cria um objeto LabelEncoder
le = LabelEncoder()

# Ajusta e transforma os dados categóricos para numéricos
dados['Occupation'] = le.fit_transform(dados['Occupation'])

# Agora, 'Workclass' é uma coluna numérica
print(dados['Occupation'].head(10))

# Salva as modificações no arquivo CSV
dados.to_csv('C:\\MineraçãoDeDados\\adult\\adultmodificacao4', index=False)


