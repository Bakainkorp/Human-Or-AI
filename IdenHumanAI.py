'''
Name: Ryan Bautista
File: IdenHumanAI.py
Date: 12/10/20

Purpose: Used as the main file

To run file
Full corpus
> python IdenHumanAI.py Corpus WriterTag.csv AIfeat.csv

Size-100 corpus
> python IdenHumanAI.py Corpus WriterTag100.csv AIfeat100.csv

Size-200 corpus
> python IdenHumanAI.py Corpus WriterTag200.csv AIfeat200.csv
'''

from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB

import nltk
import FeatureExtractor as feat
import re
import sys

# Datadir points towards the corpus, which is named "Corpus"
# Each text file in the corpus is 1 Action long, and is either Human or AI
datadir = sys.argv[1]
# WriterTag, and by extention storyprofile, stores the values as to whether the given action is human or AI
# In Corpus folder, there are three different files to be used for WriterTag: WriterTag.csv, WriterTag100.csv, and WrtierTag200.csv
storyprofile = datadir + "/" + sys.argv[2]
# File to be outputted, for exact analysis of each line
outfile = sys.argv[3]

# AuthorLabels and LoadActions preps the Actions lines in the Corpus for the program's use
def AuthorLabels():
    LineCheck = [line.rstrip().split(',') for line in open(storyprofile)]
    Storage = {row[0]:row[1] for row in LineCheck[1:]}
    return Storage
def LoadActions(Storage):
    Actions = []
    WriterID = []
    ActionID = []
    for IndividAction, IndividWriter in Storage.items():
        with open('%s/%s.txt' % (datadir, IndividAction),encoding = 'utf-8') as file:
            text = file.read()
            text = re.sub('<[^<]+?>', '', text)
            Actions.append(text)
            WriterID.append(IndividWriter)
            ActionID.append(IndividAction)
    return Actions, WriterID, ActionID

# Using the Gaussian Naive Bayers Identifiers, this is where the predictions will occur as to how accurate the program is at identifying human vs. AI text
def Prediction(Xval, Yval):
    Scores = cross_val_score(GaussianNB(), Xval, Yval, scoring='accuracy', cv=10)
    return round(Scores.mean(), 3)

# Feature Stats provides a file of all the data from each feature
def FeatureStats(Features,Header,WriterID,ActionID):
    WriterIdentif = ['Human' if g is '0' else 'AI' for g in WriterID]
    with open(outfile,'w') as file:
        file.write('ActionID\t' + '\t'.join(Header)+ '\tWriterID\n')
        for IndividAction,Line,IndividWriter in zip(ActionID,Features,WriterIdentif):
            StringLine = [str(val) for val in Line]
            file.write(IndividAction+ '\t' + '\t'.join(StringLine)+'\t'+IndividWriter+'\n')
        file.close()

if __name__ == "__main__":
    Storage = AuthorLabels()
    Actions, WriterID, ActionID = LoadActions(Storage)
    
    Features,Header = feat.FeaturePredict(Actions,2)
    print("Function Word Accuracy: " + str(Prediction(Features,WriterID)))
    Features,Header = feat.FeaturePredict(Actions,3)
    print("Syntax Accuracy: " + str(Prediction(Features,WriterID)))
    Features,Header = feat.FeaturePredict(Actions,5)
    print("Lexical Accuracy: " + str(Prediction(Features,WriterID)))
    Features,Header = feat.FeaturePredict(Actions,7)
    print("Punctuation Accuracy: " + str(Prediction(Features,WriterID)))
    Features,Header = feat.FeaturePredict(Actions,11)
    print("Complexity Accuracy: " + str(Prediction(Features,WriterID)))

    print("---------------")

    Features,Header = feat.FeaturePredict(Actions,6)
    print("Function Word + Syntax Accuracy: " + str(Prediction(Features,WriterID)))
    Features,Header = feat.FeaturePredict(Actions,10)
    print("Function Word + Lexical Accuracy: " + str(Prediction(Features,WriterID)))
    Features,Header = feat.FeaturePredict(Actions,14)
    print("Function Word + Punctuation Accuracy: " + str(Prediction(Features,WriterID)))
    Features,Header = feat.FeaturePredict(Actions,22)
    print("Function Word + Complexity Accuracy: " + str(Prediction(Features,WriterID)))

    Features,Header = feat.FeaturePredict(Actions,15)
    print("Syntax + Lexical Accuracy: " + str(Prediction(Features,WriterID)))
    Features,Header = feat.FeaturePredict(Actions,21)
    print("Syntax + Punctuation Accuracy: " + str(Prediction(Features,WriterID)))
    Features,Header = feat.FeaturePredict(Actions,33)
    print("Syntax + Complexity Accuracy: " + str(Prediction(Features,WriterID)))

    Features,Header = feat.FeaturePredict(Actions,35)
    print("Lexical + Punctuation Accuracy: " + str(Prediction(Features,WriterID)))
    Features,Header = feat.FeaturePredict(Actions,55)
    print("Lexical + Complexity Accuracy: " + str(Prediction(Features,WriterID)))

    Features,Header = feat.FeaturePredict(Actions,77)
    print("Punctuation + Complexity Accuracy: " + str(Prediction(Features,WriterID)))

    print("---------------")
    
    Features,Header = feat.FeaturePredict(Actions,30)
    print("Function Word + Syntax + Lexical Accuracy: " + str(Prediction(Features,WriterID)))
    Features,Header = feat.FeaturePredict(Actions,42)
    print("Function Word + Syntax + Punctuation Accuracy: " + str(Prediction(Features,WriterID)))
    Features,Header = feat.FeaturePredict(Actions,66)
    print("Function Word + Syntax + Complexity Accuracy: " + str(Prediction(Features,WriterID)))
    Features,Header = feat.FeaturePredict(Actions,70)
    print("Function Word + Lexical + Punctuation Accuracy: " + str(Prediction(Features,WriterID)))
    Features,Header = feat.FeaturePredict(Actions,110)
    print("Function Word + Lexical + Complexity Accuracy: " + str(Prediction(Features,WriterID)))
    Features,Header = feat.FeaturePredict(Actions,154)
    print("Function Word + Punctuation + Complexity Accuracy: " + str(Prediction(Features,WriterID)))
    
    Features,Header = feat.FeaturePredict(Actions,105)
    print("Syntax + Lexical + Punctuation Accuracy: " + str(Prediction(Features,WriterID)))
    Features,Header = feat.FeaturePredict(Actions,165)
    print("Syntax + Lexical + Complexity Accuracy: " + str(Prediction(Features,WriterID)))
    Features,Header = feat.FeaturePredict(Actions,231)
    print("Syntax + Punctuation + Complexity Accuracy: " + str(Prediction(Features,WriterID)))

    Features,Header = feat.FeaturePredict(Actions,385)
    print("Lexical + Punctuation + Complexity Accuracy: " + str(Prediction(Features,WriterID)))
    
    print("---------------")
    Features,Header = feat.FeaturePredict(Actions,1155)
    print("No Function Word Accuracy: " + str(Prediction(Features,WriterID)))
    Features,Header = feat.FeaturePredict(Actions,770)
    print("No Syntax Accuracy: " + str(Prediction(Features,WriterID)))
    Features,Header = feat.FeaturePredict(Actions,462)
    print("No Lexical Accuracy: " + str(Prediction(Features,WriterID)))
    Features,Header = feat.FeaturePredict(Actions,330)
    print("No Punctuation Accuracy: " + str(Prediction(Features,WriterID)))
    Features,Header = feat.FeaturePredict(Actions,210)
    print("No Complexity Accuracy: " + str(Prediction(Features,WriterID)))

    print("---------------")
    Features,Header = feat.FeaturePredict(Actions,2310)
    print("Full Feature Accuracy: " + str(Prediction(Features,WriterID)))
    
    print("---------------")
    print("Total number of actions analyzed: " + str(len(Actions)))
    
    FeatureStats(Features,Header,WriterID,ActionID)
'''
	print (predict_gender(features, genderlabels))
'''
