import pandas as pd

# Ler o arquivo de dados
dados = pd.read_csv('C:\\MineraçãoDeDados\\adult\\adultmodificacao4')

# Lista de colunas para calcular a média
colunas = ['Education-num', 'Marital-satus', 'Occupation', 'Relationship', 'Race', 'Capital-gain', 'Capital-loss', 'Hours-per-week', 'Native-Country', 'Income', 'Sex_Male', 'Age']

for coluna in colunas:
    # Calcular a média
    media = dados[coluna].mean()

    # Imprimir a média
    print(f"Coluna: {coluna}, Média: {media}")