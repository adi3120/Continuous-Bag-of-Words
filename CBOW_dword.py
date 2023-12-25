# import here
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
import pandas as pd
import math
from numpy.linalg import eig
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import plotly.express as px
from ANN import *

stop_words=["i","me","my","myself","we","our","ours","ourselves","you","your","yours","yourself","yourselves","he","him","his","himself","she","her","hers","herself","it","its","itself","they","them","their","theirs","themselves","what","which","who","whom","this","that","these","those","am","is","are","was","were","be","been","being","have","has","had","having","do","does","did","doing","a","an","the","and","but","if","or","because","as","until","while","of","at","by","for","with","about","against","between","into","through","during","before","after","above","below","to","from","up","down","in","out","on","off","over","under","again","further","then","once","here","there","when","where","why","how","all","any","both","each","few","more","most","other","some","such","no","nor","not","only","own","same","so","than","too","very","s","t","can","will","just","don","should","now"]

def corpus_remove_punc_symb(corpus):
  punctuations='''!()-[]{};:'"\,<>./?@#$%^&*~'''
  no_punct=""
  for char in corpus:
    if char not in punctuations:
      no_punct+=char
  return no_punct

def corpus_tokenise(corpus):
  tokens=corpus.split()
  return tokens

def corpus_remove_stop_words(corpus):
  coList=corpus_tokenise(corpus)
  for word in coList:
    if word in stop_words:
      coList.remove(word)
  return " ".join(coList)

def corpus_lemmatize(corpus):
  lemmatizer = WordNetLemmatizer()
  tokens=corpus_tokenise(corpus)
  for i in range(0,len(tokens)):
    tokens[i]=lemmatizer.lemmatize(tokens[i])
  return " ".join(tokens)


def corpus_prepare(corpus):
  corpus=corpus.lower()
  corpus=corpus_remove_punc_symb(corpus)
  corpus=corpus_remove_stop_words(corpus)
  corpus=corpus_lemmatize(corpus)
  return corpus

def vocabulary_prepare(corpus):
  vocabulary=list(set(corpus_tokenise(corpus)))
  return vocabulary

def countOccurrences(str, word):

    wordslist = list(str.split())
    return wordslist.count(word)

def co_matrix_prepare(corpus,vocabulary,window_size):
  matrix={}
  context={}
  corpus_list=corpus_tokenise(corpus)
  context_list={}
  for i in range(0,len(corpus_list)-window_size+1):
    temp_list=[]
    for j in range(0,window_size-1):
      if i+j<len(corpus_list):
        temp_list.append(corpus_list[i+j])
    context_list[tuple(temp_list)]=vocabulary.index(corpus_list[i+window_size-1])
  return context_list

  

def prepare_inputs(corpus,vocabulary,window_size):
  inputs={}
  value=np.zeros(len(vocabulary))
  corpus_list=corpus_tokenise(corpus)
  for i in range(0,len(corpus_list)-window_size+1):
    temp_list=[]
    for j in range(0,window_size-1):
      if i+j<len(corpus_list):
        temp_list.append(corpus_list[i+j])
    value=np.zeros((window_size-1)*len(vocabulary))
    for k in range(0,len(temp_list)):
      value[(k*len(vocabulary))+vocabulary.index(temp_list[k])]=1
    inputs[tuple(temp_list)]=value
  return inputs



def prepare_outputs(inputs,matrix,vocabulary):
  outputs={}
  for key in matrix.keys():
    outputs[key]=np.zeros(len(vocabulary))
    outputs[key][matrix[key]]=1
  return outputs

corpus = """Dogs are wonderful companions .
They enjoy playing fetch and running in the park.
 A well-trained dog can learn many tricks.
 Cats are independent creatures, often preferring solitude.
 Their agility and grace are admired by many.
 Some people love both dogs and cats equally, while others have a strong preference for one over the other.
 Dogs require regular exercise, whereas cats are more low-maintenance.
 Dog owners often take their pets for walks, while cats enjoy lounging indoors.
 The debate between dog lovers and cat enthusiasts is never-ending, each having valid reasons for their preferences.
 Nevertheless, both dogs and cats bring joy and comfort to countless households.
"""
window_size=4

corpus=corpus_prepare(corpus)
vocabulary=vocabulary_prepare(corpus)
co_matrix=co_matrix_prepare(corpus,vocabulary,window_size)

inputs=prepare_inputs(corpus,vocabulary,window_size)

outputs=prepare_outputs(inputs,co_matrix,vocabulary)

outputs

V=len(vocabulary)

k=30

eta=0.001

i=InputLayer((window_size-1)*V,"none")

h1=HiddenLayer(k,"none")
h1.attach_after(i)

h1_W_stacked = np.random.rand(h1.length, V)
for s in range(0,window_size-2):
  h1_Ws = np.random.rand(h1.length, V)
  h1_W_stacked = np.hstack((h1_W_stacked, h1_Ws))
  
h1.W=h1_W_stacked

h1.set_biases("zeros")

o=OutputLayer(V,"softmax","crossentropy")
o.attach_after(h1)
o.set_weights("normal_random")
o.set_biases("zeros")

ANN=[i,h1,o]


def gradient_descent_CBOW_epoch(ANN, inputs, outputs, eta, epochs):
    loss = []
    for j in range(0, epochs):
        corrects = 0
        for key in inputs.keys():
            ANN[0].put_values(inputs[key])
            ANN[len(ANN) - 1].set_actual(outputs[key])

            for layer in ANN:
                layer.forward()

            actual_label = np.argmax(outputs[key])
            output = ANN[-1].output()
            predicted_label = np.argmax(output)

            if predicted_label == actual_label:
                corrects += 1

            for i in range(len(ANN) - 1, 0, -1):
                ANN[i].backward()

            for i in range(1, len(ANN)):
                ANN[i].W -= eta * ANN[i].dLdW
                # ANN[i].Bias -= eta * ANN[i].dLda.reshape(1, -1)

        loss.append(ANN[len(ANN) - 1].loss())
        print(
            f"epoch: {j}, Loss: {ANN[len(ANN)-1].loss()}, Accuracy: {100*corrects/len(outputs)}"
        )
    return ANN, loss


ANN,loss=gradient_descent_CBOW_epoch(ANN,inputs,outputs,eta,5000)

plt.plot(loss)
plt.xlabel("epochs")
plt.ylabel("loss")
plt.show()


def plot_weight_matrices(ANN):
    num_layers = len(ANN)

    plt.figure(figsize=(15, 10))
    for i in range(1, num_layers):
        plt.subplot(2, (num_layers + 1) // 2, i)
        plt.imshow(ANN[i].W, cmap='viridis', aspect='auto')
        plt.title(f"Layer {i} Weights, Shape: {ANN[i].W.shape[0]} x {ANN[i].W.shape[1]}")
        plt.colorbar()

    plt.tight_layout()
    plt.show()

plot_weight_matrices(ANN)


testinput=('dog','wonderful','companion')
ANN[0].put_values(inputs[testinput])
for i in ANN:
  i.forward()
localoutput={}
ANNoutput=ANN[-1].output()
for i in range(0,len(vocabulary)):
  localoutput[vocabulary[i]]=ANNoutput[0][i]
sorted_output = dict(sorted(localoutput.items(), key=lambda item: item[1],reverse=True))

labels = list(sorted_output.keys())
probabilities = list(sorted_output.values())

# Plotting the probabilities for each label
plt.figure(figsize=(10, 20))
plt.barh(labels, probabilities, color='skyblue')
plt.xlabel('Probability')
plt.ylabel('Labels')
plt.title(f'Probabilities of Labels |  Input :{testinput} ')
plt.gca().invert_yaxis()  # Invert y-axis for better readability
plt.show()

def get_word_embeddings(vocabulary):
  word_embeddings={}
  for i in range(0,len(vocabulary)):
    word_embeddings[vocabulary[i]]=ANN[-1].W[i]
  word_embeddings = pd.DataFrame.from_dict(word_embeddings)

  return word_embeddings

word_embeddings=get_word_embeddings(vocabulary)

word_embeddings = word_embeddings.T

# Perform t-SNE on word embeddings to reduce dimensions to 2
tsne = TSNE(n_components=2,random_state=0)
word_embeddings_2d = tsne.fit_transform(word_embeddings)

# Plotting the 2D word embeddings
plt.figure(figsize=(12, 8))
plt.scatter(word_embeddings_2d[:, 0], word_embeddings_2d[:, 1], alpha=0.5)

# Annotate points with words
for i, word in enumerate(word_embeddings.index):
    plt.annotate(word, (word_embeddings_2d[i, 0], word_embeddings_2d[i, 1]))

plt.title('Visualization of Word Embeddings using t-SNE')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.show()



tsne_3d = TSNE(n_components=3,random_state=0)
word_embeddings_3d = tsne_3d.fit_transform(word_embeddings)

# Create a DataFrame with the 3D word embeddings
word_embeddings_3d_df = pd.DataFrame(
    {
        'x': word_embeddings_3d[:, 0],
        'y': word_embeddings_3d[:, 1],
        'z': word_embeddings_3d[:, 2],
        'word': word_embeddings.index  # Assuming your index contains words
    }
)

# Plotting the 3D word embeddings using Plotly
fig = px.scatter_3d(word_embeddings_3d_df, x='x', y='y', z='z', text='word')
fig.update_traces(textposition='top center', marker=dict(size=5))
fig.update_layout(
    title='Word Embeddings in 3D',
    scene=dict(xaxis_title='Dimension 1', yaxis_title='Dimension 2', zaxis_title='Dimension 3')
)
fig.show()


