import nltk
import numpy as np
import itertools
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from gensim.models import Word2Vec
import gudhi as g
import ripser as r


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


    lQ = list()
    lG = list()
    lL = list()

    stop_words = set(stopwords.words('spanish'))


    
    lQ = processSonnets(textQ,nQ)
    lG = processSonnets(textG,nG)
    lL = processSonnets(textL,nL)



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
    
    pcos1 = g.bottleneck_distance(dgmsG[0], dgmsQ[0])
    pcos2 = g.bottleneck_distance(dgmsG[0], dgmsL[0])
    pcos3 = g.bottleneck_distance(dgmsL[0], dgmsQ[0])
    pcos1s.append(pcos1)
    pcos2s.append(pcos2)
    pcos3s.append(pcos3)
    print("COSINE DISTANCE")
    print("Quevedo and Gongora: ",pcos1)
    print("Lope and Gongora: ",pcos2)
    print("Quevedo and Lope: ",pcos3)
    

print("MEAN COSINE DISTANCE")
print("Mean Quevedo and Gongora:",np.mean(pcos1s))
print("Mean Lope and Gongora:",np.mean(pcos2s))
print("Mean Quevedo and Lope:",np.mean(pcos3s))


np.savetxt("quevedovsgongora_dim0_cos",pcos1s)
np.savetxt("lopevsgongora_dim0_cos",pcos2s)
np.savetxt("quevedovslope_dim0_cos",pcos3s)
