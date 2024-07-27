import pandas as pd

# Ler o arquivo de dados
dados = pd.read_csv('C:\\MineraçãoDeDados\\adult\\adultmodificacao4')

# Lista de colunas para calcular o ponto médio
colunas = ['Education-num', 'Marital-satus', 'Occupation', 'Relationship', 'Race', 'Capital-gain', 'Capital-loss', 'Hours-per-week', 'Native-Country', 'Income', 'Sex_Male', 'Age']

for coluna in colunas:
    # Calcular o ponto médio
    ponto_medio = (dados[coluna].min() + dados[coluna].max()) / 2

    # Imprimir o ponto médio
    print(f"Coluna: {coluna}, Ponto Médio: {ponto_medio}")