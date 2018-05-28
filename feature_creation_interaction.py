# Data processing libraries
import pandas as pd
import numpy as np

# NLP libraries
import nltk

# Own defined functions
from drug_interaction_functions import *
from drug_functions import *
from NER_functions import *
from ortographic_features import *
from context_features import *

def sent2features(tupl, i, database):
    
    if len(tupl) != 8:
        raise ValueError('The introduced tuple does not have the correct length')
    sent = tupl[0]
    ent1 = tupl[1]
    ent2 = tupl[2]
    ent_list = tupl[3]
    pos_tags = tupl[4]
    tok_sent = tupl[5] # Tokenized sentence
    
    prefixes = r'^alk|^meth|^eth|^prop|^but|^pent|^hex|^hept|^oct|^non|^dec|^undec|^dodec|^eifcos|^di|^tri|^tetra|^penta|^hexa|^hepta'
    suffixes = r'ane$|ene$|yne$|yl$|ol$|al$|oic$|one$|ate$|amine$|amide$'

    features = {
        
    'ent1': ent1,
    'ent2': ent2,
    # Orthographic features
        
    # Entity 1
    'ent1_all_uppercase_letters' : allCaps(ent1), 
    'ent1_initial_capital_letter': initCap(ent1), 
    'ent1_contains_capital_letter' : hasCap(ent1),
    'ent1_single_capital_letter' : singleCap(ent1),
    'ent1_punctuation' : punctuation(ent1),
    'ent1_initial_digit' : initDigit(ent1),
    'ent1_single_digit' : singleDigit(ent1),
    'ent1_letter_and_num' : alphaNum(ent1),
    'ent1_many_numbers' : manyNum(ent1),
    'ent1_contains_real_numbers' : realNum(ent1),
    'ent1_intermediate_dash' : inDash(ent1),
    'ent1_has_digit' : hasDigit(ent1),
    'ent1_is_Dash' : isDash(ent1),
    'ent1_is_roman_letter' : roman(ent1),
    'ent1_is_end_punctuation' : endPunctuation(ent1),
    'ent1_caps_mix' : capsMix(ent1),

    # Entity 2
    'ent2_all_uppercase_letters' : allCaps(ent2), 
    'ent2_initial_capital_letter': initCap(ent2), 
    'ent2_contains_capital_letter' : hasCap(ent2),
    'ent2_single_capital_letter' : singleCap(ent2),
    'ent2_punctuation' : punctuation(ent2),
    'ent2_initial_digit' : initDigit(ent2),
    'ent2_single_digit' : singleDigit(ent2),
    'ent2_letter_and_num' : alphaNum(ent2),
    'ent2_many_numbers' : manyNum(ent2),
    'ent2_contains_real_numbers' : realNum(ent2),
    'ent2_intermediate_dash' : inDash(ent2),
    'ent2_has_digit' : hasDigit(ent2),
    'ent2_is_Dash' : isDash(ent2),
    'ent2_is_roman_letter' : roman(ent2),
    'ent2_is_end_punctuation' : endPunctuation(ent2),
    'ent2_caps_mix' : capsMix(ent2),
        
    # Morphological information: prefixes/suffixes of lengths from 2 to 5 and word shapes of tokens. 
    # Entity 1
    'ent1_word[-5:]': ent1[-5:],
    'ent1_word[-4:]': ent1[-4:],
    'ent1_word[-3:]': ent1[-3:],
    'ent1_word[-2:]': ent1[-2:],

    # Entity 2
    'ent2_word[-5:]': ent2[-5:],
    'ent2_word[-4:]': ent2[-4:],
    'ent2_word[-3:]': ent2[-3:],
    'ent2_word[-2:]': ent2[-2:],
    
    # Domain knowledge
    # Entity 1
    'ent1_drug_sufix': getSuffix(ent1, suffixes),
    'ent1_drug_prefix': getPrefix(ent1, prefixes),

    # Entity 2
    'ent2_drug_sufix': getSuffix(ent2, suffixes),
    'ent2_drug_prefix': getPrefix(ent2, prefixes),

        
    # Is in DrugBank dataset
    'ent1_isInDB':isTokenInDB(ent1,database),
    'ent2_isInDB':isTokenInDB(ent2,database),
    
        
    # Context features
    'n_tokens_bw_entities': countTokensBetweenEntities(tok_sent, ent1, ent2),
    'n_entities_bw_entities': countEntitiesBetweenEntities(tok_sent, ent1, ent2, ent_list),
    'n_modal_verbs_bw_entities': countModalVerbsBetweenEntities(tok_sent, ent1, ent2),
    'sentence_contains_neg': sentenceContainsNegation(tok_sent),
    'keywords_bw_entities': keyWordsBetweenEntities(tok_sent, ent1, ent2),
    'first_modal_sentence': getFirstModalVerb(tok_sent),
    'POS_tags_sentence_simpl': createSimplifiedPOSPath(pos_tags),
    '2_grams_bw_entities': getNgramsBetweenEntities(tok_sent, ent1, ent2, 2),
    '3_grams_bw_entities': getNgramsBetweenEntities(tok_sent, ent1, ent2, 3)
        
    }

    return features


def text2features(text,database):
    for i in range(len(text)):
        return(sent2features(text, i, database))

def text2interaction(text):
    return text[6]

def text2interactionType(text):
    return text[-1]