'''
Name: Ryan Bautista
File: FeatureExtractor.py
Date: 12/10/20

Purpose: Used for the individual features for analysis
'''

import nltk
from nltk.corpus import stopwords
import re
import numpy as nump
from collections import Counter

# Keeps track of all Functions Words
# Function words include pronouns, adverbs, and other words that don't mean anything in particular on their own, but mean something within the context of a sentence
def FunctionWords(texts):
    Bow = []
    Header = stopwords.words('english')
    for text in texts:
        counts = []
        tokens = nltk.word_tokenize(text)
        for sw in stopwords.words('english'):
            sw_count = tokens.count(sw)
            normed = round(sw_count/float(len(tokens)) , 3)
            counts.append(normed)
        Bow.append(counts)
    Bow_nump = nump.array(Bow).astype(float)
    return Bow_nump, Header

# Keeps track of the syntax tags of a sentence
def Syntax(texts):
    Bow = []
    SyntaxTags = ['NN', 'NNP', 'DT', 'IN', 'JJ', 'NNS','CC','PRP','VB','VBG']
    Header = SyntaxTags
    for text in texts:
        counts = []
        tokens = nltk.word_tokenize(text)
        tagged = nltk.pos_tag(tokens)
        S_tags = [t[-1] for t in tagged]
        for pos in Header:
            sw_count = S_tags.count(pos)
            normed = round(sw_count/float(len(S_tags)),3)
            counts.append(normed)
        Bow.append(counts)
    Bow_nump = nump.array(Bow).astype(float)
    return Bow_nump, Header

# Keeps track of the number of the most recurring words
def Lexical(texts):
    Bow = []
    Header = []
    TokenList = []
    for text in texts:
        tokens = nltk.word_tokenize(text)
        for token in tokens:
            TokenList.append(token)
    AllCounts = Counter(TokenList)
    Commons = AllCounts.most_common(5)
    for w in Commons:
        Header.append(w[0])
    for text in texts:
        counts = []
        tokens = nltk.word_tokenize(text)
        for word in Header:
            counts.append(tokens.count(word))
        Bow.append(counts)
    Bow_nump = nump.array(Bow)
    return Bow_nump, Header

# Keeps track of punctuation
def Punctuation(texts):
    Bow = []
    PunctuationChars = ['.',',',':','-','\'','\"','(','!','?']
    Header = PunctuationChars
    for text in texts:
        counts = []
        tokens = nltk.word_tokenize(text)
        for p in PunctuationChars:
            p_count = tokens.count(p)
            normed = round(p_count/float(len(tokens)),3)
            counts.append(normed)
        Bow.append(counts)
    Bow_nump = nump.array(Bow).astype(float)
    return Bow_nump, Header

# Tracks the complexity of the words, number of long words, and the length of each sentence
def Complexity(texts):
    Bow = []
    Header = ["Char/Word","UniWord","Word/Sent","LongWord"]
    for text in texts:
        tokens = nltk.word_tokenize(text)
        WordLength = 0
        for word in tokens:
            WordLength += len(word)
        CharsPerWord = round(WordLength/len(tokens), 3)
        UniqueWord = 0
        for word in tokens:
            if tokens.count(word) == 1:
                UniqueWord += 1
        UniqueWordFreq = round(UniqueWord/len(tokens), 3)
        Sentences = nltk.sent_tokenize(text)
        SentenceLength = 0
        for sentence in Sentences:
            SentenceLength += len(sentence)
        WordsPerSentence = round(SentenceLength/len(Sentences), 3)
        LongWord = 0
        for word in tokens:
            if len(word) >=6:
                LongWord += 1
        counts = [CharsPerWord,UniqueWordFreq,WordsPerSentence,LongWord]
        Bow.append(counts)
    Bow_nump = nump.array(Bow).astype(float)
    return Bow_nump, Header


def FeaturePredict(texts, modNumber):
    Features = []
    Headers = []

    # Each call to Feature predict includes a number divisible by certain numbers. If the number included is evenly divisible, that feature will be tested
    # Every combination of feature and features will be tested for their accuracy, and the results will be collated into the paper
    if(modNumber % 2 == 0):
        f,h = FunctionWords(texts)
        Features.append(f)
        Headers.extend(h)

    if(modNumber % 3 == 0):
        f,h = Syntax(texts)
        Features.append(f)
        Headers.extend(h)

    if(modNumber % 5 == 0):
        f,h = Lexical(texts)
        Features.append(f)
        Headers.extend(h)

    if(modNumber % 7 == 0):
        f,h = Punctuation(texts)
        Features.append(f)
        Headers.extend(h)

    if(modNumber % 11 == 0):
        f,h = Complexity(texts)
        Features.append(f)
        Headers.extend(h)

    CheckedFeatures = nump.concatenate(Features,axis=1)
    return CheckedFeatures, Headers
