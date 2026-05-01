# Importa as bibliotecas necessárias para baixar dados (kagglehub) e manipulá-los (pandas).
import kagglehub
import pandas as pd

# Baixa o dataset Iris do Kaggle e o carrega em um DataFrame.
path = kagglehub.dataset_download("uciml/iris")
print("Path to dataset files:", path)

# -------------------------------------------------------------------------------------------------------#

class Node():
  # Construtor da classe Node, representando um nó na árvore de decisão.
  # Guarda a característica de divisão, o limite, a impureza Gini, e os nós filhos.
  # Se for uma folha, guarda o valor da classe prevista.
  def __init__(self, feature=None, threshold=None, gini=None, left=None, right=None, valor=None):
    self.feature = feature
    self.threshold = threshold
    self.gini = gini
    self.left = left
    self.right = right
    self.valor = valor

# -------------------------------------------------------------------------------------------------------#

def gini(df):
  # Calcula a impureza de Gini para um DataFrame.
  # Mede a probabilidade de um elemento ser incorretamente classificado.
  size_df = df.shape[0]
  sentosa, versicolor, virginica = 0, 0, 0

  if size_df == 0: return [0, None] # Gini é 0 se o DataFrame estiver vazio.

  # Conta as ocorrências de cada espécie.
  for row in df['Species']:
    if row == "Iris-setosa": sentosa += 1
    elif row == "Iris-versicolor": versicolor += 1
    elif row == "Iris-virginica": virginica += 1

  # Fórmula de Gini: 1 - soma((P_i)^2).
  gini_df = 1 - (sentosa/size_df)**2 - (versicolor/size_df)**2 - (virginica/size_df)**2

  return [size_df, gini_df]

# -------------------------------------------------------------------------------------------------------#

def split(df):
  # Encontra a melhor característica e limite para dividir o DataFrame.
  # O objetivo é minimizar a impureza de Gini resultante da divisão.

  features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
  best_feature = ""
  best_gini = 1 # Gini inicial (máxima impureza).
  threshold = 0

  # Itera por cada característica para testar as divisões.
  for feature in features:
    max_value = int(max(df[feature]) * 100)
    min_value = int(min(df[feature]) * 100)
    gini_min = 1
    fi_min = 0

    # Itera sobre possíveis thresholds (pontos de corte) para a feature atual.
    for i in range(min_value + 5, max_value - 5, 10):
      fi = i/100 # Converte de volta para float.
      left = df[df[feature] < fi]
      right = df[df[feature] >= fi]

      left_size, gini_left = gini(left)
      right_size, gini_right = gini(right)

      if left_size == 0 or right_size == 0: continue # Ignora divisões vazias.

      total = left_size + right_size
      # Calcula o Gini ponderado pela proporção dos subconjuntos.
      gini_now = (gini_left/total)*left_size + (gini_right/total)*right_size

      # Atualiza o melhor Gini e threshold para a feature atual.
      if gini_now < gini_min:
        gini_min = gini_now
        fi_min = fi

    # Compara o melhor Gini da feature atual com o melhor Gini geral encontrado.
    if (gini_min < best_gini):
      best_gini = gini_min
      best_feature = feature
      threshold = fi_min

  # Retorna a melhor característica, seu limite, o Gini e os DFs divididos.
  return [best_feature, threshold, best_gini, df[df[best_feature] < threshold], df[df[best_feature] >= threshold]]

# -------------------------------------------------------------------------------------------------------#

def majoritary_class(df):
  # Determina a classe (espécie) mais frequente em um DataFrame.
  # Usado para definir a previsão de nós folha.

  sentosa, versicolor, virginica = 0, 0, 0
  classes = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
  classe = 0

  # Conta as ocorrências de cada espécie.
  for row in df['Species']:
    if row == "Iris-setosa": sentosa += 1
    elif row == "Iris-versicolor": versicolor += 1
    elif row == "Iris-virginica": virginica += 1

  # Encontra a classe com a maior contagem.
  if sentosa >= versicolor and sentosa >= virginica: classe = 0
  elif versicolor >= sentosa and versicolor >= virginica: classe = 1
  elif virginica >= sentosa and virginica >= versicolor: classe = 2

  return classes[classe] # Retorna o nome da classe majoritária.

# -------------------------------------------------------------------------------------------------------#

def create_tree(df):
  # Função recursiva para construir a árvore de decisão.

  # Condições de parada: nó puro (Gini 0) ou número mínimo de amostras (<=5).
  if (gini(df)[1] == 0 or len(df) <= 5):
    majority = majoritary_class(df) # Encontra a classe majoritária para o nó folha.
    return Node(valor = majority) # Retorna um nó folha.

  # Tenta dividir o DataFrame.
  best_feature, limiar, best_gini, left, right = split(df)

  # Se a divisão resultar em subconjuntos vazios, o nó se torna uma folha.
  if (len(left) == 0 or len(right) == 0):
    majority = majoritary_class(df)
    return Node(valor = majority)

  # Constrói recursivamente os nós filhos esquerdo e direito.
  left_way = create_tree(left)
  right_way = create_tree(right)

  # Retorna o nó atual com a melhor divisão e seus filhos.
  return Node(feature = best_feature, threshold = limiar, gini = best_gini, left = left_way, right = right_way)

# -------------------------------------------------------------------------------------------------------#

def predict(root, data):
  # Função para fazer uma previsão usando a árvore de decisão.
  # root: nó atual da árvore; data: amostra a ser classificada.

  if root.valor is not None:
    return root.valor # Se for um nó folha, retorna a classe prevista.
  
  # Se não for folha, compara o valor da característica com o threshold para descer na árvore.
  if data[root.feature] < root.threshold:
    return predict(root.left, data) # Desce pelo caminho esquerdo.
  else:
    return predict(root.right, data) # Desce pelo caminho direito.

# -------------------------------------------------------------------------------------------------------#

# Carrega em um dataframe o dataset importado no início
df = pd.read_csv(path + "/Iris.csv")

# Embaralha o DataFrame para garantir uma divisão aleatória dos dados.
df = df.sample(frac=1)

# Define o número de folds (partições) para a validação cruzada.
S = 10
acuracia = 0

# Loop para realizar a validação cruzada S vezes.
for i in range(S):
  # Define o bloco de teste (15 amostras por fold).
  bl_test = df.iloc[i*15:(i+1)*15,:]
  # Concatena as partes antes e depois do bloco de teste para formar o conjunto de treino.
  bl_train1 = df.iloc[:i*15,:]
  bl_train2 = df.iloc[(i+1)*15:,:]
  bl_train = pd.concat([bl_train1, bl_train2])

  # Constrói a árvore de decisão com os dados de treino do fold atual.
  root = create_tree(bl_train)

  # Avalia o modelo no conjunto de teste do fold atual.
  ok = 0 # Reinicia a contagem de acertos para cada fold.
  for index, row in bl_test.iterrows():
    # Faz a previsão e compara com a classe real.
    if predict(root, row) == row['Species']:
      ok += 1 # Conta acertos.
  
  # Acumula a acurácia para este fold. A divisão por 'S' é para calcular a média posteriormente.
  acuracia += ok / (len(bl_test)*S)

print(f"Acurácia: {acuracia}")

# -------------------------------------------------------------------------------------------------------#
