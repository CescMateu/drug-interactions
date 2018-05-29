
from NER_functions import *


def word2features(sent, i, database,freqDistribution,rare_freq):
    word = sent[i][0]
    postag = sent[i][1]
    
    # orthographic features
    features = {
    'all_uppercase_letters' : allCaps(word), 
    'initial_capital_letter': initCap(word), 
    'contains_capital_letter' : hasCap(word),
    'single_capital_letter' : singleCap(word),
    'punctuation' : punctuation(word),
    'initial_digit' : initDigit(word),
    'single_digit' : singleDigit(word),
    'letter_and_num' : alphaNum(word),
    'many_numbers' : manyNum(word),
    'contains_real_numbers' : realNum(word),
    'intermediate_dash' : inDash(word),
    'has_digit' : hasDigit(word),
    'is_Dash' : isDash(word),
    'is_roman_letter' : roman(word),
    'is_end_punctuation' : endPunctuation(word),
    'caps_mix' : capsMix(word),
        
    # Morphological information: prefixes/suffixes of lengths from 2 to 5 and word shapes of tokens. 
    'word[-5:]': word[-5:],
    'word[-4:]': word[-4:],    
    'word[-3:]': word[-3:],
    'word[-2:]': word[-2:],
    
    # Domain knowledge
    'contains_drug_sufix': containsSufix(word),
    'contains_drug_prefix': containsPrefix(word),

    # POS tag   
    'postag': postag,
    'postag[:2]': postag[:2],
        
    # Is in DrugBank dataset
    'isInDB':isTokenInDB(word,database),

    # freq of the token
    'freq': freqDistribution[word],
    'rare_word': isRareWord(word,freqDistribution,rare_freq)
    }
    
    
    
    # context features
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
            '-1:unigrams' : num_ngrams(word1,1),
            '-1:bigrams' : num_ngrams(word1,2),
            '-1:trigrams' : num_ngrams(word1,3)
        })
        if i!=1:
            word2 = sent[i-2][0]
            postag2 = sent[i-2][1]
            features.update({
            '-2:postag': postag2,
            '-2:postag[:2]': postag2[:2],
            '-2:unigrams' : num_ngrams(word2,1)})
            if len(word2)>2:
                features.update({
                    '-2:bigrams' : num_ngrams(word2,2),
                    '-2:trigrams' : num_ngrams(word2,3)})
            elif len(word2)==2:
                features.update({
                    '-2:bigrams' : num_ngrams(word2,2)})
                
            
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
            '+1:unigrams' : num_ngrams(word1,1),
            '+1:bigrams' : num_ngrams(word1,2),
            '+1:trigrams' : num_ngrams(word1,3)
        })
        
        if i<len(sent)-2:
            word2 = sent[i+2][0]
            postag2 = sent[i+2][1]
            features.update({
                '+2:postag': postag2,
                '+2:postag[:2]': postag2[:2],
                '+2:unigrams' : num_ngrams(word2,1)})
            if len(word2)>2:
                features.update({
                    '+2:bigrams' : num_ngrams(word2,2),
                    '+2:trigrams' : num_ngrams(word2,3)})
            elif len(word2)==2:
                features.update({
                    '+2:bigrams' : num_ngrams(word2,2)})
            
    else:
        features['EOS'] = True

    return features


def sent2features(sent,database,freqDistribution,rare_freq):
    return [word2features(sent, i,database,freqDistribution,rare_freq) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, postag, label in sent]

def sent2tokens(sent):
    return [token for token, postag, label in sent]