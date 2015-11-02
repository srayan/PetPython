import nltk
import nltk.classify.util
import nltk.metrics
from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
import string
stopwords = nltk.corpus.stopwords.words('english')

lemma = WordNetLemmatizer()
def word_feats(words):
    return dict([(word, True) for word in words])

def word_feats_WithoutStopwords(words):
    return dict([(word, True) for word in words if word not in stopwords])

def word_feats_Tokens(words):
    return dict([(word, True) for word in words if word not in stopwords and word not in string.punctuation])

def word_feats_Lemma(words):
    word_dictionary = dict()
    for word in words:
        if word not in stopwords and word not in string.punctuation:
            if(word.endswith("ing")):
                lword = lemma.lemmatize(word,"v")
            else:
                lword = lemma.lemmatize(word,"n")
            word_dictionary[lword] = True
    return word_dictionary



def maxEntClassifier(negFeats,posFeats,negCut,posCut):
    trainDataSet = negFeats[:negCut] + posFeats[:posCut]
    testDataSet = negFeats[negCut:] + posFeats[posCut:]

    maxEntAlgorithm = nltk.classify.MaxentClassifier.ALGORITHMS[0]
    maxEntClassifier = nltk.MaxentClassifier.train(trainDataSet, maxEntAlgorithm, max_iter = 10)

    maxEntClassifier.show_most_informative_features(10)

    print 'train accuracy:', nltk.classify.util.accuracy(maxEntClassifier, trainDataSet)
    print 'test accuracy:', nltk.classify.util.accuracy(maxEntClassifier, testDataSet)

print 'Number of english Stop Words',len(stopwords)

negFileIds = movie_reviews.fileids('neg')
posFileIds = movie_reviews.fileids('pos')

negFeatsWithStopwords = [(word_feats(movie_reviews.words(fileids=[f])), 'neg') for f in negFileIds]
posFeatsWithStopwords = [(word_feats(movie_reviews.words(fileids=[f])), 'pos') for f in posFileIds]
#Pos=Neg
negCutWithStopwords = len(negFeatsWithStopwords) * 3/4
posCutWithStopwords = len(posFeatsWithStopwords) * 3/4
#Pos<Neg
negCutWithStopwords2 = len(negFeatsWithStopwords) * 3/4
posCutWithStopwords2 = len(posFeatsWithStopwords) * 1/2
#Pos>Neg
negCutWithStopwords3 = len(negFeatsWithStopwords) * 1/2
posCutWithStopwords3 = len(posFeatsWithStopwords) * 3/4


negFeatsWithoutStopwords = [(word_feats_WithoutStopwords(movie_reviews.words(fileids=[f])), 'neg') for f in negFileIds]
posFeatsWithoutStopwords = [(word_feats_WithoutStopwords(movie_reviews.words(fileids=[f])), 'pos') for f in posFileIds]
#Pos=Neg
negCutWithoutStopwords = len(negFeatsWithoutStopwords) * 3/4
posCutWithoutStopwords = len(posFeatsWithoutStopwords) * 3/4
#Pos < Neg
negCutWithoutStopwords2 = len(negFeatsWithoutStopwords) * 3/4
posCutWithoutStopwords2 = len(posFeatsWithoutStopwords) * 1/2
#Pos > Neg
negCutWithoutStopwords3 = len(negFeatsWithoutStopwords) * 1/2
posCutWithoutStopwords3 = len(posFeatsWithoutStopwords) * 3/4

negFeatsTokens = [(word_feats_Tokens(movie_reviews.words(fileids=[f])), 'neg') for f in negFileIds]
posFeatsTokens = [(word_feats_Tokens(movie_reviews.words(fileids=[f])), 'pos') for f in posFileIds]
#Pos = neg
negCutTokens = len(negFeatsTokens) * 3/4
posCutTokens = len(posFeatsTokens) * 3/4
#Pos < neg
negCutTokens2 = len(negFeatsTokens) * 3/4
posCutTokens2 = len(posFeatsTokens) * 1/2
#Pos > neg
negCutTokens3 = len(negFeatsTokens) * 1/2
posCutTokens3 = len(posFeatsTokens) * 3/4

negFeatsLemma = [(word_feats_Lemma(movie_reviews.words(fileids=[f])), 'neg') for f in negFileIds]
posFeatsLemma = [(word_feats_Lemma(movie_reviews.words(fileids=[f])), 'pos') for f in posFileIds]
#Pos=Neg
negCutLemma = len(negFeatsLemma) * 3/4
posCutLemma = len(posFeatsLemma) * 3/4
#Pos<Neg
negCutLemma2 = len(negFeatsLemma) * 3/4
posCutLemma2 = len(posFeatsLemma) * 1/2
#Pos>Neg
negCutLemma3 = len(negFeatsLemma) * 1/2
posCutLemma3 = len(posFeatsLemma) * 3/4

print("\n################################# Pos = Neg ################################################")

print('\n******************************* Including Stop Words **************************************')
maxEntClassifier(negFeatsWithStopwords,posFeatsWithStopwords,negCutWithStopwords,posCutWithStopwords)

print('\n******************************* Removal of Stop Words **************************************')
maxEntClassifier(negFeatsWithoutStopwords,posFeatsWithoutStopwords,negCutWithoutStopwords,posCutWithoutStopwords)

print('\n******************************* Removal of Stop Words & Punctuation  **************************************')
maxEntClassifier(negFeatsTokens,posFeatsTokens,negCutTokens,posCutTokens)

print('\n******************************* Removal of Stop Words ; Punctuation  ; Lemmatized words **************************************')
maxEntClassifier(negFeatsLemma,posFeatsLemma,negCutLemma,posCutLemma)

print("\n -------------------------------------------------------------------------------------------------------------------------------")
print("\n################################# Pos < Neg ################################################")
maxEntClassifier(negFeatsWithStopwords,posFeatsWithStopwords,negCutWithStopwords2,posCutWithStopwords2)

print('\n******************************* Removal of Stop Words **************************************')
maxEntClassifier(negFeatsWithoutStopwords,posFeatsWithoutStopwords,negCutWithoutStopwords2,posCutWithoutStopwords2)

print('\n******************************* Removal of Stop Words & Punctuation  **************************************')
maxEntClassifier(negFeatsTokens,posFeatsTokens,negCutTokens2,posCutTokens2)

print('\n******************************* Removal of Stop Words ; Punctuation  ; Lemmatized words **************************************')
maxEntClassifier(negFeatsLemma,posFeatsLemma,negCutLemma2,posCutLemma2)


print("\n -------------------------------------------------------------------------------------------------------------------------------")
print("\n################################# Pos > Neg ################################################")
maxEntClassifier(negFeatsWithStopwords,posFeatsWithStopwords,negCutWithStopwords3,posCutWithStopwords3)

print('\n******************************* Removal of Stop Words **************************************')
maxEntClassifier(negFeatsWithoutStopwords,posFeatsWithoutStopwords,negCutWithoutStopwords3,posCutWithoutStopwords3)

print('\n******************************* Removal of Stop Words & Punctuation  **************************************')
maxEntClassifier(negFeatsTokens,posFeatsTokens,negCutTokens3,posCutTokens3)

print('\n******************************* Removal of Stop Words ; Punctuation  ; Lemmatized words **************************************')
maxEntClassifier(negFeatsLemma,posFeatsLemma,negCutLemma3,posCutLemma3)
