import numpy as np
import random

# [0] - номер документа
# [1] - номер слова
# [2] - сколько раз слово в документе

def get_doc_data(docName):
  params = {
    'test1.dat': {
      'T': 3, 'alpha': 1, 'beta': 1,
    },
    'test2.dat': {
      'T': 20, 'alpha': 0.1, 'beta': 0.1,
    }
  }

  documents = np.loadtxt(docName, dtype=int)

  return [documents, params[docName], 100 if docName == 'test1.dat' else 2]

[docdata, params, iterationsCount] = get_doc_data('test1.dat')
uniqueWordsCount = max(map(lambda x: x[1], docdata))
docsCount = max(map(lambda x: x[0], docdata))


NWT = [[0 for j in range(0, params['T'])] for i in range(0, uniqueWordsCount)]
NTD = [[0 for j in range(0, docsCount)] for i in range(0, params['T'])]
NT = [0 for j in range(0, params['T'])]

word_topic = {}

# Инициализация
for data_row in docdata:
  [doc_num, word_num, word_num_count_in_doc] = data_row
  # Чтобы поместились в массив
  [doc_num, word_num] = [doc_num - 1, word_num - 1]

  for i in range(0, word_num_count_in_doc):
    topic_number = random.randint(0, params['T'] - 1)

    # print(f"word_num: {word_num}, topic_number: {topic_number}")
    NWT[word_num][topic_number] += 1
    NTD[topic_number][doc_num] += 1
    NT[topic_number] += 1

    word_topic[f"{doc_num}-{word_num}-{i}"] = topic_number


# Схема Collapsed Gibbs sampling

for _ in range(0, iterationsCount):
  for data_row in docdata:
    [doc_num, word_num, word_num_count_in_doc] = data_row
    # Чтобы поместились в массив
    [doc_num, word_num] = [doc_num - 1, word_num - 1]

    # Итерируемся по каждому слову в отдельности
    for i in range(0, word_num_count_in_doc):
      topic_number = word_topic.get(f"{doc_num}-{word_num}-{i}")

      NWT[word_num][topic_number] -= 1; NTD[topic_number][doc_num] -= 1; NT[topic_number] -= 1;

      P = []
      for t in range(0, params['T']):
        Pt = (params['alpha'] + NTD[t][doc_num]) * (params['beta'] + NWT[word_num][t]) / params['beta'] * uniqueWordsCount + NT[t]
        Pt = max(0, Pt)
        P.append(Pt)

      sumP = sum(P)
      P = P / sumP

      new_topic_number = np.random.choice(range(0, params['T']), p=P)
      word_topic[f"{doc_num}-{word_num}-{i}"] = new_topic_number
      NWT[word_num][new_topic_number] += 1; NTD[new_topic_number][doc_num] += 1; NT[new_topic_number] += 1;

print(NT)