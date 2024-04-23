import pandas as pd

# Suponha que df é o seu DataFrame e 'Age' é a coluna que você quer analisar

# Calcule a frequência de cada valor
freq = df['Age'].value_counts()

# Calcule a frequência relativa de cada valor
relative_freq = df['Age'].value_counts(normalize=True)

# Crie uma tabela com os valores, frequências e frequências relativas
freq_table = pd.DataFrame({'Valor': freq.index, 'Frequência': freq.values, 'Frequência Relativa': relative_freq.values})

# Exiba a tabela
print(freq_table)