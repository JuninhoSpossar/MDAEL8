import pandas as pd

# Função para categorizar a educação
def categorize_education(education):
    fundamental = ["5th-6th", "1st-4th", "7th-8th", "Preschool"]
    medio = ["9th", "10th", "11th", "12th"]
    superior = ["Bachelors", "Some-college", "HS-grad", "Prof-school", "Assoc-acdm", "Assoc-voc", "Masters", "Doctorate"]

    if education in fundamental:
        return "Fundamental"
    elif education in medio:
        return "Médio"
    elif education in superior:
        return "Superior"
    else:
        return "Outro"  # Caso existam classes não mapeadas

# Leitura do arquivo CSV
input_file = 'C:\\MineraçãoDeDados\\adult\\adult.csv'
df = pd.read_csv(input_file) 

#Coluna que categorize education em fundamental, médio, superior e outro
df["Education2"] = df["Education"].apply(categorize_education)


print(df)
print(df.columns)
