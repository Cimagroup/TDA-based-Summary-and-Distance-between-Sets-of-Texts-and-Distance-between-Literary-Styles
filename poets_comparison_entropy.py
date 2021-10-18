import nltk
nltk.download('stopwords')
nltk.download('punkt')
import numpy as np
import itertools
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from gensim.models import Word2Vec
import gudhi as g
import ripser as r
from persim.persistent_entropy import *


stemmer = SnowballStemmer('spanish')
textQ = 'Quevedo/Quevedo_'
textG = 'Gongora/Gongora_'
textL = 'LopeDeVega_1/LopeDeVega_'

pcos1s = list()
pcos2s = list()
pcos3s = list()

def processSonnets(texts,n):
    l2 = list()
    for i in n:
        textos = open(texts+str(i)+'.txt')
        poema = textos.read()
        sentences = [p for p in poema.split('\n') if p]
        l = list()
        for i in range(len(sentences)):
            tokens = nltk.word_tokenize(sentences[i])
            l.append(tokens)
        k = list()
        for i in range(len(l)):
            words = [stemmer.stem(word.lower()) for word in l[i] if word.isalpha() and(not word.lower() in stop_words)]
            k.append(words)
        l2.append(k)
    return l2


for i in range(100):

    nQ = range(1,116)
    nG = range(1,116) 
    nL = range(1,116)
    nL2 =range(117,125)
    nG2 = range(117,125)
    nQ2=range(117,125)

    lQ = list()
    lG = list()
    lL = list()
    lQ2 =list()
    lL2 = list()

    stop_words = set(stopwords.words('spanish'))


    
    lQ = processSonnets(textQ,nQ)
    lG = processSonnets(textG,nG)
    lL = processSonnets(textL,nL)
    #lG2 = processSonnets(textG,nG2)
    #lL2 = processSonnets(textL,nL2)
    



    mergedG = list(itertools.chain(*lG))
    mergedL = list(itertools.chain(*lL))
    mergedQ = list(itertools.chain(*lQ))
    data = mergedG+mergedL+mergedQ
    w2v = Word2Vec(data, size = 150, window = 10, iter = 250)
    word_vectors = w2v.wv
    vocab_embedding = word_vectors.vocab.keys()
    vocabularyQ = list(itertools.chain.from_iterable(lQ))
    vocabularyQ = list(itertools.chain.from_iterable(vocabularyQ))

    vocabularyG = list(itertools.chain.from_iterable(lG))
    vocabularyG = list(itertools.chain.from_iterable(vocabularyG))

    vocabularyL = list(itertools.chain.from_iterable(lL))
    vocabularyL = list(itertools.chain.from_iterable(vocabularyL))

    vocabulary = set(vocabularyQ+vocabularyG+vocabularyL)
    Quevedo_embedding = np.stack([word_vectors[word] for word in vocab_embedding if word in vocabularyQ])
    Gongora_embedding = np.stack([word_vectors[word] for word in vocab_embedding if word in vocabularyG])
    Lope_embedding    = np.stack([word_vectors[word] for word in vocab_embedding if word in vocabularyL])
    k = np.min([len(Quevedo_embedding),len(Gongora_embedding),len(Lope_embedding)])
    Quevedo_embedding = Quevedo_embedding[0:k]
    Gongora_embedding = Gongora_embedding[0:k]
    Lope_embedding    = Lope_embedding[0:k]
    
    dgmsQ = r.ripser(Quevedo_embedding, metric = "cosine")['dgms']
    dgmsG = r.ripser(Gongora_embedding, metric = "cosine")['dgms']
    dgmsL = r.ripser(Lope_embedding, metric = "cosine")['dgms']
    
    num_bars = np.min([len(dgmsQ[0]),len(dgmsL[0]),len(dgmsL[1])])
    
    pcos1 = persistent_entropy(dgmsQ[0][-num_bars:])
    pcos2 = persistent_entropy(dgmsL[0][-num_bars:])
    pcos3 = persistent_entropy(dgmsG[0][-num_bars:])
    pcos1s.append(pcos1)
    pcos2s.append(pcos2)
    pcos3s.append(pcos3)
    print("Persistent entropy")
    print("Quevedo: ",pcos1)
    print("Lope: ",pcos2)
    print("Quevedo: ",pcos3)
    

print("MEAN PERSISTENT ENTROPY")
print("Mean Quevedo and Gongora:",np.mean(pcos1s))
print("Mean Lope and Gongora:",np.mean(pcos2s))
print("Mean Quevedo and Lope:",np.mean(pcos3s))


np.savetxt("persistent_entropy_quevedo",pcos1s)
np.savetxt("persistent_entropy_lope",pcos2s)
np.savetxt("persistent_entropy_gongora",pcos3s)
