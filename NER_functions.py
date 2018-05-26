# functions needed for the NER_classifier

import nltk,warnings,re
import pandas as pd
from nltk.tag import StanfordPOSTagger
from operator import itemgetter # this will be needed when sorting a list of tuples
import string

from binary_features_functions import *

# Define precision and recall function for model evaluation

def compute_recall(pred_ent,true_ent):
    if len(true_ent)==0: 
        if len(pred_ent)!=0: return 0
        else: return -1
    else: return round(len([word for word in pred_ent if word in true_ent])/len(true_ent),2)*100


def compute_precision(pred_ent,true_ent):
    if len(true_ent)==0: 
        if len(pred_ent)!=0: return 0
        else: return -1
    else:
        if len(pred_ent)==0: return -1
        else: return round(len([word for word in pred_ent if word in true_ent])/len(pred_ent),2)*100     

# Orthographic features from paper
def initCap(string):
    if re.match('^[A-Z].*',string):
        return 1
    else: return 0
    
def allCaps(string):
    if re.match('^[A-Z]+$',string):
        return 1
    else: return 0

def hasCap(string):
    if re.match('^.*[A-Z].*$',string):
        return 1
    else: return 0
    
def singleCap(string):
    if re.match('^[A-Z]$',string):
        return 1
    else: return 0

def punctuation(string):
    if re.match('^[,;:\'\"]$',string):
        return 1
    else: return 0
    
def initDigit(string):
    if re.match('^[0-9].*',string):
        return 1
    else: return 0

def singleDigit(string):
    if re.match('^[0-9]$',string):
        return 1
    else: return 0

def alphaNum(string):
    if re.match('.*[A-Za-z].*[0-9].*|.*[0-9].*[A-Za-z].*',string):
        return 1
    else: return 0
    
def manyNum(string):
    if re.match('^[0-9]{1,2}(,[0-9]{1,2})+$',string):
        return 1
    else: return 0
    
def realNum(string):
    if re.match('^-?[0-9]+[\.][0-9]+$',string):
        return 1
    else: return 0 
    
def inDash(string): #intermidiate dash
    if re.match('^([\w+][\-]+)+\w+$',string):
        return 1
    else: return 0
    
def hasDigit(string): 
    if re.match('.*[0-9].*',string):
        return 1
    else: return 0
    
def isDash(string): 
    if re.match('^[-]+$',string):
        return 1
    else: return 0
    
def roman(string): 
    if re.match('^[IVXDLCM]+$',string):
        return 1
    else: return 0

def endPunctuation(string): 
    if re.match('^[.?!]$',string):
        return 1
    else: return 0

def capsMix(string): 
    if re.match('.*[A-Z].*[a-z].*|.*[a-z].*[A-Z].*',string):
        return 1
    else: return 0

# -----------------------------------------------------------------
def num_ngrams(input_string,n):
    ngrams = []
    while len(input_string)>=n:
        ngrams.append(input_string[:n])
        input_string=input_string[1:]
    return ngrams

# -----------------------------------------------------------------
def containsSufix(word):
    
    
    # Self-implemented
    sufixes = r'ane$|ene$|yne$|ol$|al$|amine$|cid$|ium$|ether$|ate$|afil$|asone$|bicin$|bital$|caine$|cillin$|cycline$|dazole$|dipine$|dronate$|eprazole$|fenac$|floxacin$|gliptin$|glitazone$|iramine$|lamide$|mab$|mustine$|mycin$|nacin$|nazole$|olol$|olone$|onide$|oprazole$|parin$|phylline$|pramine$|pril$|profen$|ridone$|sartan$|semide$|setron$|slatin$|tadine$|terol$|thiazide$|tinib$|trel$|tretin$|triptan$|tyline$|vir$|vudine$|zepam$|zodone$|zolam$|zosin$'
    
    
    # sufixes = r'ane$|ene$|yne$|yl$|ol$|al$|oic$|one$|ate$|amine$|amide$'
    
    if re.search(sufixes,word): return 1
    else: return 0

def containsPrefix(word):
    
    '''
    Self-implemented
    prefixes = r'^meth|^eth|^prop|^but|^pent|^hex|^hept|^oct|^non|^dec|^cef|^sulfa|^ceph|^pred'
    '''
    
    prefixes = r'^alk|^meth|^eth|^prop|^but|^pent|^hex|^hept|^oct|^non|^dec|^undec|^dodec|^eifcos|^di|^tri|^tetra|^penta|^hexa|^hepta'
    
    if re.search(prefixes,word): return 1
    else: return 0


# -----------------------------------------------------------------
# BIO TAGGER

def BIOTagger(text, drugs):
    ''' (str, list of str) -> list of tuples
    
    Description: 
    Given a sentence 'text' and a set of drugs 'drugs', this function returns a list of str that
    contains a tag for each of the tokens in text. The tags can be either 'B', 'I' or 'O'. 'B' means
    the token is the first part of a drug entity, 'I' means the token is the continuation of a drug entity,
    and 'O' means that the token does not belong to a drug entity.
    
    Examples/Tests:
    
    >>> BIOTagger('Ibuprofeno is great!', ['Ibuprofeno'])
    [('Ibuprofeno', 'B'), ('is', 'O'), ('great', 'O'), ('!', 'O')]
    >>> BIOTagger('I would like to buy calcium-rich milk', ['calcium-rich'])
    [('I', 'O'), ('would', 'O'), ('like', 'O'), ('to', 'O'), ('buy', 'O'), ('calcium-rich', 'B'), ('milk', 'O')]
    >>> BIOTagger('Give me TNF antioxidants together with sodium, please', ['TNF antioxidants', 'sodium'])
    [('Give', 'O'), ('me', 'O'), ('TNF', 'B'), ('antioxidants', 'I'), ('together', 'O'), ('with', 'O'), ('sodium', 'B'), (',', 'O'), ('please', 'O')]
    >>> BIOTagger('Give me TNF antioxidants together with sodium, please I want Iboprufeno Dalci', ['TNF antioxidants', 'sodium','Iboprufeno','Dalci'])
    [('Give', 'O'), ('me', 'O'), ('TNF', 'B'), ('antioxidants', 'I'), ('together', 'O'), ('with', 'O'), ('sodium', 'B'), (',', 'O'), ('please', 'O'), ('I', 'O'), ('want', 'O'), ('Iboprufeno', 'B'), ('Dalci', 'B')]
    
    '''

    # Preprocessing
    # Tokenize all the words and elements of the original sentence
    tokens = nltk.word_tokenize(text)
    
    # change the format of drugs to make them fit with a regular expression
    drugs_one_word = []
    for drug in drugs:
        if ' ' not in drug[0]: # we ignore those drug entities with multiple words
            drugs_one_word.append(drug)
    
    # creating a list of those drug entities comprised of more than one word and, again, changing its format
    drugs_multiple_words = [(drug[0].split(),drug[1]) for drug in drugs if ' ' in drug[0]]
    print('drugs multiples: ',drugs_multiple_words)
    # initializing some parameters and the list of tags to be returned
    tokens_to_skip = 0
    bio_tagged = []
    for token in tokens:
        token_tagged = False
        if tokens_to_skip !=0:
            tokens_to_skip -=1
        else: 
            for drug in drugs_one_word: 
                if token == drug[0]:
                    bio_tagged.append('B-'+drug[1]) # remember drug has the following shape: ('Ibuprofeno','drug')
                    token_tagged = True
                    break
                    
                    
            # if token does not match with some one-word drug, let's see if it is the first word of a multi-word drug entity
            if not token_tagged:
                for drug in drugs_multiple_words:
                    if token == drug[0][0]:
                        bio_tagged.append('B-'+drug[1])
                        bio_tagged = bio_tagged + ['I']*(len(drug[0])-1)
                        tokens_to_skip = len(drug)-1
                        token_tagged = True
                        break
                if not token_tagged:
                    bio_tagged.append('O')
                
    bio_tagged = list(zip(tokens,bio_tagged))
    print(bio_tagged)
    return(bio_tagged)

# -----------------------------------------------------------------
# BIO TAGS TO ENTITIES

def BIOTagsToEntities(tokens, bio_tags):
    '''
    Description:
    Given a list of tokens, 'tokens' and a list of tags for each token, 'bio_tags', this function returns a list of all the entities
    detected that had a 'B' or a 'I' associated. This function will be used in order to retrieve the entities predicted by the classification
    model and compare them with the real ones. 

    Examples/Tests:
    >>> BIOTagsToEntities(tokens = ['START', 'Ibuprofeno', 'is', 'good', '.'], bio_tags = ['O', 'B', 'O', 'O', 'O'])
    ['Ibuprofeno']
    >>> BIOTagsToEntities(tokens = ['START', 'Food', 'is', 'good', '.'], bio_tags = [])
    []
    >>> BIOTagsToEntities(tokens = ['START', 'TNF', 'Receptors', 'are', 'good', '.', 'STOP'], bio_tags = ['O', 'B', 'I', 'O', 'O', 'O', 'O'])
    ['TNF Receptors']
    >>> BIOTagsToEntities(tokens = ['START', 'TNF', 'Receptors', 'and', 'Fluimicil', '400mg', 'are', 'good', '.', 'STOP'], bio_tags = ['O', 'B', 'I', 'O', 'B', 'I', 'O', 'O', 'O', 'O'])
    ['TNF Receptors', 'Fluimicil 400mg']
    >>> BIOTagsToEntities(tokens = ['START', 'TNF', 'Receptors', 'and', 'Fluimicil', '400mg', 'are', 'good', '.', 'STOP'], bio_tags = ['O', 'B', 'I', 'B', 'I', 'I', 'O', 'O', 'O', 'O'])
    ['TNF Receptors', 'and Fluimicil 400mg']
    '''

    # Initialise the necessary objects
    entities = []
    prev_tag = 'O'
    word = ''

    # Iterate over all the bio_tags and tokens and retrieve those ones that have a 'B' or 'I' associated 
    for idx in range(0, len(bio_tags)):
        tag = bio_tags[idx]

        if tag == 'B':
            if prev_tag in ['B','I']:
                # If a 'B' tag is found and the previous one was either a 'B' or a 'I', we
                # append the previous entity to the list of entities
                entities.append(word)
            word = tokens[idx]
            prev_tag = 'B'

        elif tag == 'I' and prev_tag in ['B','I']:
            # If a 'I' tag is found, we need to update the current entity (it is formed by more than one word)
            word = word + ' ' + tokens[idx]
            prev_tag = 'I'

        elif tag == 'O' and prev_tag in ['B','I']:
            # 'If a 'O' tag is found and the previous tag was either a 'B' or a 'I', 
            # we append the previous entity to the list of entities (same cases than the first if condition)
            entities.append(word)
            prev_tag = 'O'

        elif tag == 'O' and prev_tag == 'O':
            # Do nothing, no entity appended and go to the next tag
            continue
        else:
            # Any other case should raise an error
            warnings.warn('One of the tags was not recognised. Please check the "bio_tags" parameter.')
    
    return entities

# -----------------------------------------------------------------
# IS TOKEN IN DRUGBANK DATASET
def isTokenInDB(token, db_list):
    ''' (string, list) -> bool
    This function checks if a word 'token' is inside a given DB 'db_list' of drugs that has been
    extracted from the DrugBank database

    >>> isTokenInDB('Ibuprofeno', ['Ibuprofeno', 'Almax'])
    True
    >>> isTokenInDB('Ibuprofeno', ['Fluimucil', 'Paracetamol'])
    False
    '''

    return(token in db_list)
# -----------------------------------------------------------------
# -----------------------------------------------------------------

if __name__ == '__main__':
    import doctest
    doctest.testmod()

