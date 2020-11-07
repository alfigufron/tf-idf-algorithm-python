import nltk
import pandas as pd
import numpy as np
from math import log

def tf(doc, wordBank):
  wordList = doc.str.split(' ')
  # Mengubah isi data doc menjadi array

  maxFt = [len(s)  for s in wordList]
  # Menghitung jumlah data pada variabel wordList

  result = pd.DataFrame()
  for index, word in wordBank.iterrows():
    # Dilakukan perulangan untuk kata untuk menjalankan rumus tf
    ft = np.add([s.count(word['words']) for s in wordList], 0)
    ftd = 1 + np.log10(ft + 1e-15)
    result = result.append(pd.Series(ftd), ignore_index=True)

  result = result.replace(-np.inf, 0)

  return result

def idf(doc, wordBank):
  wordList = doc.str.split(' ')

  result = pd.DataFrame(columns=['word', 'idf'])
  for index, word in wordBank.iterrows():
    # Dilakukan perulangan untuk kata untuk menjalankan rumus idf
    dft = np.add([s.count(word['words']) for s in wordList], 0)
    idft = log(1 / dft)
    result = result.append(pd.Series([word['words'],  idft], index=['word', 'idf']), ignore_index=True)

  return result

def tfIdf(tf, idf):
  result = pd.DataFrame()
  for i in tf:
    # Dilakukan perulangan dari data tf untuk menjalankan rumus tfidf
    tfIdf = tf[i] * idf['idf']
    result = result.append(pd.Series(tfIdf), ignore_index=True)

  return result

def tfDigram(doc, wordBank):
  maxFt = [len(s)  for s in doc]
  # Menghitung jumlah data yang ada

  result = pd.DataFrame()
  for index, word in wordBank.iterrows():
    # Dilakukan perulangan untuk kata untuk menjalankan rumus tf
    ft = np.add([s.count(word['words']) for s in doc], 0)
    if ft > 0:
      ftd = 1 + np.log10(1)
    else:
      ftd = 1 + np.log10(0)

    result = result.append(pd.Series(ftd), ignore_index=True)

  result = result.replace(-np.inf, 0)

  return result

def idfDigram(doc, wordBank):
  result = pd.DataFrame(columns=['word', 'idf'])
  # Dilakukan perulangan untuk kata untuk menjalankan rumus idf
  for index,word in wordBank.iterrows():
    if word['words'] in doc[0]:
      dft = np.sum(1)
    else:
      dft = np.sum(0)
    idft = log(1 / dft)
    result = result.append(pd.Series([word['words'],  idft], index=['word', 'idf']), ignore_index=True)

  return result

def tfIdfDigram(tf, idf):
  result = pd.DataFrame()
  for i in tf:
    # Dilakukan perulangan dari data tf untuk menjalankan rumus tfidf
    tfIdf = tf[i] * idf['idf']
    result = result.append(pd.Series(tfIdf), ignore_index=True)

  return result

# -------- TF IDF --------

df_csv = pd.read_csv('data/file.csv', encoding='utf-8')
# Mengambil data dari file csv, lalu mengubah ke format dataframe

text = ""
for index, row in df_csv.iterrows():
  # Dibuat perulangan untuk mengambil data perbaris dari dataframe lalu dibuat menjadi satu baris teks
  text += row['Stopword'].replace('[','').replace("'","").replace(",","").replace("]"," ")

data = {'sentences':(text)}
doc = pd.DataFrame([data])
doc = doc['sentences']
# Hasil teks yang didapatkan diubah menjadi dataframe dalam satu variabel

itemText = []
listText = list(text.split(' '))
del listText[-1]
# Hasil teks yang didapatkan diubah menjadi array dalam satu variabel
# Menghapus data array paling terakhir karena hanya berisi spasi

for item in listText:
  if item in itemText:
    pass
  else:
    itemText.append(item)
# Dibuat perulangan untuk mengambil konteks kata agar tidak ada kesamaan kata pada data konteks

data = {'words':itemText}
vocabulary = pd.DataFrame(data=data)
# Data konteks yang sudah didapatkan dari proses perulangan diubah menjadi dataframe

resultTF = tf(doc, vocabulary)
# Memasukan kalimat dan data konteks ke fungsi tf

resultIDF = idf(doc, vocabulary)
# Memasukan kalimat dan data konteks ke fungsi idf

resultTfIdf = tfIdf(resultTF, resultIDF)
resultTfIdf.columns = itemText
resultTfIdf = resultTfIdf.transpose()
# Memasukan hasil dari fungsi tf dan hasil dari fungsi idf ke fungsi tfidf

resultTfIdf.to_csv('result.csv', encoding='utf-8')
# Mengubah format dataframe yang di hasilkan menjadi format csv



# -------- TF IDF Digram --------

tokenizeDoc = nltk.word_tokenize(text)
bigramDoc = nltk.bigrams(tokenizeDoc)
listBigramDoc = list(bigramDoc)
# Menentukan digram pada data text

arrBigramDoc = []
for item in listBigramDoc:
  # Mengubah data digram menjadi array
  itemStrDoc = ' '.join(item)
  arrBigramDoc.append(itemStrDoc)

data = {'sentences':(arrBigramDoc)}
doc = pd.DataFrame([data])
doc = doc['sentences']
# Hasil array digram diubah menjadi dataframe pada satu variabel

itemDigram = []
for item in arrBigramDoc:
  # Mencari data konteks dan membuatnya menjadi array pada suatu variabel
  if item in itemDigram:
    pass
  else:
    itemDigram.append(item)

data = {'words':itemDigram}
vocabulary = pd.DataFrame(data=data)
# Data konteks digram yang sudah di dapatkan diubah menjadi dataframe

resultTfDigram = tfDigram(doc, vocabulary)
# Memasukan Digram dan konteks Digram ke fungsi tf

resultIdfDigram = idfDigram(doc, vocabulary)
# Memasukan Digram dan konteks Digram ke fungsi idf

resultTfIdfDigram = tfIdfDigram(resultTfDigram, resultIdfDigram)
resultTfIdfDigram.columns = itemDigram
resultTfIdfDigram = resultTfIdfDigram.transpose()
# Memasukan hasil dari fungsi tfDigram dan hasil dari fungsi idfDigram ke fungsi tfidf

resultTfIdfDigram.to_csv('resultDigram.csv', encoding='utf-8')
# Mengubah format dataframe yang di hasilkan menjadi format csv