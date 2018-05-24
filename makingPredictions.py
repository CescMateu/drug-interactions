
import nltk, statistics
import pandas as pd
from drug_functions import *

def makingPredictions(texts_entities,clf,drugbank_db,st):#,training_dummies):
    predictions = []
    for text,entities in texts_entities:
        #print('text: ', text)
        #print('real entities: ',entities,'\n')
        
        # tokenize text
        tokens = nltk.word_tokenize(text)
        # computing predictions
        features = createFeatureVector(tokens, drugbank_db,st)
        
        
        '''
        # computing one-hot coding for 'Aa1-' feature.
        dummies = pd.get_dummies(features['Aa1-'])
        features = features.drop('Aa1-',axis=1)
        # joining both data frames
        for name in dummies.columns:
            features[name]=dummies[name]

        #print(tokens)
        
        # adding those columns related to Aa1- that we cannot see with the sentence in question
        for name in training_dummies.columns:
            if name not in dummies.columns:
                features[name]=[0]*len(dummies[dummies.columns.values[0]])
        
        '''
        
        predicted_tags = clf.predict(features)
        
        
        predictions.append((list(predicted_tags),entities,text)) 
        #print('predicted bio tags: ',predicted_tags,'\n')
        #pred_entities = BOTagsToEntities(tokens = tokens, bo_tags = predicted_tags)
        #print('predicted entities: ', pred_entities, '\n')
        
    # predictions is a list of tupples comprised of predicted tags and the true drugs we should extract from there
    #print('predictions of text 1: ',predictions[1])

    precision = []
    recall = []
    for tags, true_entities, text in predictions:
        # I need the tokens for the bioTagsToEntities function
        tokens = nltk.word_tokenize(text)
        predicted_entities = BIOTagsToEntities(tokens,tags)
        precision = precision + [compute_precision(predicted_entities,true_entities)]
        recall = recall + [compute_recall(predicted_entities,true_entities)]

        
    avg_precision = statistics.mean(precision)
    avg_recall = statistics.mean(recall)


    # F1 metric
    F1 = round((2*avg_precision*avg_recall) / (avg_precision + avg_recall),2)

    return F1,avg_precision,avg_recall