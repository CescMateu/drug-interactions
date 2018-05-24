import nltk,warnings,re
import pandas as pd
from nltk.tag import StanfordPOSTagger
from operator import itemgetter # this will be needed when sorting a list of tuples

from binary_features_functions import *

# Define precision and recall function for model evaluation
def compute_recall(pred_ent,true_ent):
    if len(pred_ent) == 0 or len(true_ent) == 0: return 0
    else: return round(len([word for word in pred_ent if word in true_ent])/len(true_ent),2)*100


def compute_precision(pred_ent,true_ent):
    if len(pred_ent) == 0 or len(true_ent) == 0: return 0
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

# we could unify the three folowing functions, but I leave it like this for the seek of a better understandability
def num_unigrams(input_string):
    n = 1
    ngrams = []
    while len(input_string)>=n:
        ngrams.append(input_string[:n])
        input_string=input_string[1:]
    return len(ngrams)
def num_bigrams(input_string):
    n = 2
    ngrams = []
    while len(input_string)>=n:
        ngrams.append(input_string[:n])
        input_string=input_string[1:]
    return len(ngrams)
def num_trigrams(input_string):
    n = 3
    ngrams = []
    while len(input_string)>=n:
        ngrams.append(input_string[:n])
        input_string=input_string[1:]
    return len(ngrams)

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
    >>> BIOTagger('I would like to buy calcium-rich milk', ['calcium'])
    [('I', 'O'), ('would', 'O'), ('like', 'O'), ('to', 'O'), ('buy', 'O'), ('calcium-rich', 'B'), ('milk', 'O')]
    >>> BIOTagger('Give me TNF antioxidants together with sodium, please', ['TNF antioxidants', 'sodium'])
    [('Give', 'O'), ('me', 'O'), ('TNF', 'B'), ('antioxidants', 'I'), ('together', 'O'), ('with', 'O'), ('sodium', 'B'), (',', 'O'), ('please', 'O')]
    '''

    # Preprocessing
    # Tokenize all the words and elements of the original sentence
    tokens = nltk.word_tokenize(text)
    # Separate all the drugs into individual words
    drugs = sum([word.split() for word in drugs],[])

    # Bio Tagger
    # Initialise the bio_tagged list that will accumulate the results and
    # the 'prev_tag' variable
    bio_tagged = []
    prev_tag = 'O'

    # For each token of the original sentence, we will check whether there is an entity contained in the token
    # We will follow a very simple rules to tag each token with the 'BIO' system
    for token in tokens:
        if prev_tag == 'O' : # Begin NE or continue O

            # In here we contemplate two different cases: 
            # - Case 1: The token is exactly the same as one of the drugs list
            # - Case 2: The token has a substring equal to a drug ('calcium-rich' case)
            if any([drug in token for drug in drugs]):
                bio_tagged.append((token,'B'))
                prev_tag = 'B'
            else:
                bio_tagged.append((token,'O'))
                prev_tag = 'O'

        elif prev_tag == 'B': # Inside NE

            if any([drug in token for drug in drugs]):
                bio_tagged.append((token,'I'))
                prev_tag = 'I'
            else: 
                bio_tagged.append((token,'O'))
                prev_tag = 'O'

        elif  prev_tag == 'I': # Inside NE

            if any([drug in token for drug in drugs]):
                bio_tagged.append((token,'I'))
                prev_tag = 'I'
            else: 
                bio_tagged.append((token,'O'))
                prev_tag = 'O'
        
    return(bio_tagged)

def splitDashTokens(tokens):
    '''
    >>> splitDashTokens(['hola-que', 'hola', 'no-vull-res'])
    ['hola', 'que', 'hola', 'no', 'vull', 'res']
    '''

    split_tokens = []
    for token in tokens:
        if '-' in token:
            to_append = token.split('-')
            for token_part in to_append:
                split_tokens.append(token_part)
        else:
            split_tokens.append(token)

    return(split_tokens)

def BOTagger(text, drugs):
    ''' (str, list of str) -> list of tuples
    
    Description: 
    Very similar to BIOTagger(), but with just two classes: 'B' and 'O'. Given a sentence 'text' and a set of drugs 'drugs', this function returns a list of str that
    contains a tag for each of the tokens in text. The tags can be either 'B', or 'O'. 'B' means
    the token is the first part of a drug entity and 'O' means that the token does not belong to a drug entity.
    
    Examples/Tests:
    
    >>> BOTagger('Ibuprofeno is great!', ['Ibuprofeno'])
    [('Ibuprofeno', 'B'), ('is', 'O'), ('great', 'O'), ('!', 'O')]
    >>> BOTagger('I would like to buy calcium-rich milk', ['calcium'])
    [('I', 'O'), ('would', 'O'), ('like', 'O'), ('to', 'O'), ('buy', 'O'), ('calcium', 'B'), ('rich', 'O'), ('milk', 'O')]
    >>> BOTagger('Give me TNF antioxidants together with sodium, please', ['TNF antioxidants', 'sodium'])
    [('Give', 'O'), ('me', 'O'), ('TNF', 'B'), ('antioxidants', 'B'), ('together', 'O'), ('with', 'O'), ('sodium', 'B'), (',', 'O'), ('please', 'O')]
    >>> BOTagger('Give me TNF antioxidants together with Exter diodorant sodium, please', ['TNF antioxidants', 'Exter diodorant sodium'])
    [('Give', 'O'), ('me', 'O'), ('TNF', 'B'), ('antioxidants', 'B'), ('together', 'O'), ('with', 'O'), ('Exter', 'B'), ('diodorant', 'B'), ('sodium', 'B'), (',', 'O'), ('please', 'O')]
    >>> BOTagger('Ibu-Fluimi is good.', ['Ibu-Fluimi'])
    [('Ibu', 'B'), ('Fluimi', 'B'), ('is', 'O'), ('good', 'O'), ('.', 'O')]
    '''

    ## Initialise the different lists that we need
    # Tokenize the original text and split the dashed words
    tokens = nltk.word_tokenize(text)
    #tokens = splitDashTokens(tokens)

    # Also separate the drugs with dashes
    #drugs = splitDashTokens(drugs)
    # Create a list with the initial word of each drug entity
    first_drug = [drug.split()[0] for drug in drugs]
    # Create a list with the length of each drug entity
    ent_length = [len(drug.split()) for drug in drugs]
    # Create a list with the tokens that are already tagged
    already_tagged = [0]*len(tokens)
    
    # Initialise the list in which to accumulate the tags
    BO_tags = []

    ## BO Tagger
    # Iterate over all the tokens in the list
    for idx_t, t in enumerate(tokens):
        if already_tagged[idx_t] == 0:
            # We havent tagged the token yet
            if t in first_drug:
                # Recover the index of the entity
                idx_ent = first_drug.index(t)
                # Create a list with as many B's as the length of the entity
                new_B_tags = ['B'] * ent_length[idx_ent]
                # Attach the new B's to the result
                BO_tags.extend(new_B_tags)
                # Update the already_tagged list
                for i in range(ent_length[idx_ent]):
                    already_tagged[idx_t + i] = 1
            else:
                BO_tags.append('O')
                already_tagged[idx_t] = 1

        else:
            # We have already tagged that token
            continue

    if(len(tokens) != len(BO_tags)):
        print(tokens, BO_tags)
        stop('Lenghts of the two lists do not coincide')
    
    return(list(zip(tokens, BO_tags)))



def checkPreviousTokenCondition(tokens, pos, condition):
    '''(list of str, int, function) -> list of bool
    Description:
    Given a list of tokens 'tokens' to be analyzed and an integer 'pos', this function returns a 
    list of booleans indicating, for each token in the list, if the token in the position +- 'pos'
    complies with the condition specified in 'condition'. This function introduces an exception: If the token
    analyzed is either 'START' or 'STOP', it won't be processed and will return 0 in all cases. Also, is 'START'
    and 'STOP' are the tokens in the 'pos' position in respect the token analized, also a 0 will be returned.
    
    Requirements:
    - The function in 'condition' needs to be a function that gets an string as an input an returns a boolean
    as an output in 0/1 format, not False/True.
    - 'pos' can only take one of the following values: (-2, -1, 0, 1, 2)
    
    Examples/Tests:
    >>> tokens_example = ['START', 'Ibuprofeno', 'is', 'good']
    >>> def isUpperCap(string):
    ...      return int(string.isupper())
    
    >>> checkPreviousTokenCondition(tokens = tokens_example, pos = -2, condition = isUpperCap)
    [0, 0, 0, 0]
    
    >>> tokens_example = ['START', 'My', 'name', 'Is', 'Cesc', 'Mateu', '.']
    >>> def hasInitialUpperCase(string):
    ...       return int(string[0].isupper())
    >>> checkPreviousTokenCondition(tokens = tokens_example, pos = -1, condition = hasInitialUpperCase)
    [0, 0, 1, 0, 1, 1, 1]
    
    >>> tokens_example = ['START', 'My', 'name', 'Is', 'Cesc', 'Mateu', 'Jove', 'STOP', 'START', 'How', 'ARE', 'you', '?', 'STOP', 'START']
    >>> def hasInitialUpperCase(string):
    ...       return int(string[0].isupper())
    >>> checkPreviousTokenCondition(tokens = tokens_example, pos = -1, condition = hasInitialUpperCase)
    [0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0]
    
    >>> tokens_example = ['START', 'My', 'name', 'Is', 'Cesc', 'Mateu', '.']
    >>> def hasInitialUpperCase(string):
    ...       return int(string[0].isupper())
    >>> checkPreviousTokenCondition(tokens = tokens_example, pos = 0, condition = hasInitialUpperCase)
    [0, 1, 0, 1, 1, 1, 0]
    
    >>> tokens_example = ['START', 'My', 'name', 'Is', 'Cesc', 'Mateu', '.']
    >>> def hasInitialUpperCase(string):
    ...       return int(string[0].isupper())
    >>> checkPreviousTokenCondition(tokens = tokens_example, pos = 2, condition = hasInitialUpperCase)
    [0, 1, 1, 1, 0, 0, 0]
    '''
    
    # Specify the range in which to check for the condition depending on 'pos'
    if pos <= 0:
        range_var = range(abs(pos), len(tokens))
        # Initialize the resulting boolean list correctly
        boolean_list = [False] * abs(pos)
    else:
        range_var = range(0, len(tokens) - abs(pos))
        boolean_list = []
    
    # Iterate over all the tokens in the list, and check the previous/following tokens
    # for the specified condition
    
    for idx_token in range_var:
        if tokens[idx_token + pos] == 'START' or tokens[idx_token + pos] == 'STOP':
            boolean_list.append(False)
        elif tokens[idx_token] == 'START' or tokens[idx_token] == 'STOP':
            boolean_list.append(False)
        else:
            boolean_list.append(condition(tokens[idx_token + pos]))
           
    # If we are checking the posterior elements of a list, at the end we will need to append 0's
    if pos > 0:
        boolean_list = boolean_list + [False] * abs(pos)
    
    return([int(element) for element in boolean_list])

def BOTagsToEntities(tokens, bo_tags): 
    ''' 
    Examples/Tests:
    >>> BOTagsToEntities(tokens = ['START', 'Ibuprofeno', 'is', 'not','good', '.'], bo_tags = ['O', 'B', 'O', 'B', 'B', 'O'])
    ['Ibuprofeno', 'not good']
    >>> BOTagsToEntities(tokens = ['START', 'TNF', 'Receptors', 'are', 'good', '.', 'STOP'], bo_tags = ['O', 'B', 'B', 'O', 'O', 'B', 'O'])
    ['TNF Receptors', '.']
    
    '''
    
    entities =[]
    prev_tag = 'O'
    word = ''
    tokens_tags = zip(tokens,bo_tags)
    #print('tokens_tags ',list(tokens_tags))
    for token,tag in tokens_tags:
        if tag =='B':
            if prev_tag == 'B':
                word = word + ' ' + token
                prev_tag='B'
            elif prev_tag=='O':
                word = token
                prev_tag = 'B'
        elif tag=='O':
            if prev_tag=='B':
                entities.append(word)
                prev_tag='O'
            
    return entities


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
    for idx in range(0, len(bio_tags)-1):
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


# Define a function for the automatized creation of features given a tokenized sentence

def createFeatureVector(tokenized_sentence, drugbank_db,st):
    ''' (string, list) -> dict
    Description:
    
    Examples/Tests:
    
    '''
    # tokenized_sentence = nltk.word_tokenize(text)

    # Feature: Initialise the feature_vector data frame, in which we will create the features of each token
    feature_vector = pd.DataFrame()
    
    # Feature: Length of the token
    feature_vector['token_length'] = [len(token) for token in tokenized_sentence]
    
    # Feature: Prefixes and Suffixes

    prefix_feature = []
    suffix_feature = []
    '''
    Self-implemented
    prefixes = r'^meth|^eth|^prop|^but|^pent|^hex|^hept|^oct|^non|^dec|^cef|^sulfa|^ceph|^pred'
    suffixes = r'ane$|ene$|yne$|ol$|al$|amine$|cid$|ium$|ether$|ate$|afil$|asone$|bicin$|bital$|caine$|cillin$|cycline$|dazole$|dipine$|dronate$|eprazole$|fenac$|floxacin$|gliptin$|glitazone$|iramine$|lamide$|mab$|mustine$|mycin$|nacin$|nazole$|olol$|olone$|onide$|oprazole$|parin$|phylline$|pramine$|pril$|profen$|ridone$|sartan$|semide$|setron$|slatin$|tadine$|terol$|thiazide$|tinib$|trel$|tretin$|triptan$|tyline$|vir$|vudine$|zepam$|zodone$|zolam$|zosin$'
    '''
    
    prefixes = r'^alk|^meth|^eth|^prop|^but|^pent|^hex|^hept|^oct|^non|^dec|^undec|^dodec|^eifcos|^di|^tri|^tetra|^penta|^hexa|^hepta'
    suffixes = r'ane$|ene$|yne$|yl$|ol$|al$|oic$|one$|ate$|amine$|amide$'
    for token in tokenized_sentence:

            if re.search(prefixes,token):
                prefix_feature=prefix_feature+[1]
            else:
                prefix_feature = prefix_feature+[0]

            if re.search(suffixes,token):
                suffix_feature=suffix_feature+[1]
            else:
                suffix_feature = suffix_feature+[0]

    feature_vector['prefix_feature']=prefix_feature
    feature_vector['suffix_feature']=suffix_feature

    
    # Feature: unigrams, bigrams and trigrams of POS (POS) in window of [-2, 2]
    
    
    # tuples = st.tag(tokenized_sentence)
    tuples = nltk.pos_tag(tokenized_sentence)
    pos = [word[1] for word in tuples]

    # initializing context lists
    unigram_pos_minus_2 = []
    unigram_pos_minus_1 = []
    unigram_pos_0 = []
    unigram_pos_1 = []
    unigram_pos_2 = []

    bigram_pos_minus_2 = []
    bigram_pos_minus_1 = []
    bigram_pos_0 = []
    bigram_pos_1 = []
    bigram_pos_2 = []

    trigram_pos_minus_2 = []
    trigram_pos_minus_1 = []
    trigram_pos_0 = []
    trigram_pos_1 = []
    trigram_pos_2 = []
    
    for i in range(0,len(tokenized_sentence)):
        if i in [0,1]:
            unigram_pos_minus_2.append(0)
            bigram_pos_minus_2.append(0)
            trigram_pos_minus_2.append(0)
        else:
            unigram_pos_minus_2.append(num_unigrams(pos[i-2]))
            bigram_pos_minus_2.append(num_bigrams(pos[i-2]))
            trigram_pos_minus_2.append(num_trigrams(pos[i-2]))
        
        if i==0:
            unigram_pos_minus_1.append(0)
            bigram_pos_minus_1.append(0)
            trigram_pos_minus_1.append(0)
        else:
            unigram_pos_minus_1.append(num_unigrams(pos[i-1]))
            bigram_pos_minus_1.append(num_bigrams(pos[i-1]))
            trigram_pos_minus_1.append(num_trigrams(pos[i-1]))
        
        unigram_pos_0.append(num_unigrams(pos[i]))
        bigram_pos_0.append(num_bigrams(pos[i]))
        trigram_pos_0.append(num_trigrams(pos[i]))
        
        if i != len(tokenized_sentence)-1:
            unigram_pos_1.append(num_unigrams(pos[i+1]))
            bigram_pos_1.append(num_bigrams(pos[i+1]))
            trigram_pos_1.append(num_trigrams(pos[i+1]))
        else: 
            unigram_pos_1.append(0)
            bigram_pos_1.append(0)
            trigram_pos_1.append(0)
                                 
        if i not in [len(tokenized_sentence)-2,len(tokenized_sentence)-1]:
            unigram_pos_2.append(num_unigrams(pos[i+2]))
            bigram_pos_2.append(num_bigrams(pos[i+2]))
            trigram_pos_2.append(num_trigrams(pos[i+2]))
        else:
            unigram_pos_2.append(0)
            bigram_pos_2.append(0)
            trigram_pos_2.append(0)
    
    
    # adding the feature lists to the data frame
    feature_vector['unigram_pos_minus_2'] = unigram_pos_minus_2
    feature_vector['bigram_pos_minus_2'] = bigram_pos_minus_2
    feature_vector['trigram_pos_minus_2'] = trigram_pos_minus_2
    feature_vector['unigram_pos_minus_1'] = unigram_pos_minus_1
    feature_vector['bigram_pos_minus_1'] = bigram_pos_minus_1
    feature_vector['trigram_pos_minus_1'] = trigram_pos_minus_1
    feature_vector['unigram_pos_0'] = unigram_pos_0
    feature_vector['bigram_pos_0'] = bigram_pos_0
    feature_vector['trigram_pos_0'] = trigram_pos_0
    feature_vector['unigram_pos_1'] = unigram_pos_1
    feature_vector['bigram_pos_1'] = bigram_pos_1
    feature_vector['trigram_pos_1'] = trigram_pos_1
    feature_vector['unigram_pos_2'] = unigram_pos_2
    feature_vector['bigram_pos_2'] = bigram_pos_2
    feature_vector['trigram_pos_2'] = trigram_pos_2
    
    
    # Feature: Binary token type orthographic features

    feature_vector['all_uppercase_letters'] = [allCaps(token) for token in tokenized_sentence]
    feature_vector['initial_capital_letter'] = [initCap(token) for token in tokenized_sentence]
    feature_vector['contains_capital_letter'] = [hasCap(token) for token in tokenized_sentence]
    feature_vector['single_capital_letter'] = [singleCap(token) for token in tokenized_sentence]
    feature_vector['punctuation'] = [punctuation(token) for token in tokenized_sentence]
    feature_vector['initial_digit'] = [initDigit(token) for token in tokenized_sentence]
    feature_vector['single_digit'] = [singleDigit(token) for token in tokenized_sentence]
    feature_vector['letter_and_num'] = [alphaNum(token) for token in tokenized_sentence]
    feature_vector['many_numbers'] = [manyNum(token) for token in tokenized_sentence]
    feature_vector['contains_real_numbers'] = [realNum(token) for token in tokenized_sentence]
    feature_vector['intermediate_dash'] = [inDash(token) for token in tokenized_sentence]
    feature_vector['has_digit'] = [hasDigit(token) for token in tokenized_sentence]
    feature_vector['is_Dash'] = [isDash(token) for token in tokenized_sentence]
    feature_vector['is_roman_letter'] = [roman(token) for token in tokenized_sentence]
    feature_vector['is_end_punctuation'] = [endPunctuation(token) for token in tokenized_sentence]
    feature_vector['caps_mix'] = [capsMix(token) for token in tokenized_sentence]
    
    
    # Feature: Is the token the first/last of the sentence?

    is_first_word_sentence = []

    first_token = False
    for token in tokenized_sentence:
        if token == 'START':
            first_token = True
            is_first_word_sentence.append(0)
        elif first_token == True:
            is_first_word_sentence.append(1)
            first_token = False
        elif first_token == False:
            is_first_word_sentence.append(0)
    

    feature_vector['is_first_word_sentence'] = is_first_word_sentence


    
    
    # Feature: Binary token type features of the +-2 previous/following tokens
    
    for k in [-2, -1, 0, 1, 2]:
        '''
        all_uppercase_letters_context = checkPreviousTokenCondition(tokens = tokenized_sentence, 
                                                                 pos = k, condition = allUpperCase)
        all_lowercase_letters_context = checkPreviousTokenCondition(tokens = tokenized_sentence, 
                                                                 pos = k, condition = allLowerCase)
        initial_capital_letter_context = checkPreviousTokenCondition(tokens = tokenized_sentence, 
                                                                 pos = k, condition = hasInitialCapital)
        contains_slash_context = checkPreviousTokenCondition(tokens = tokenized_sentence, 
                                                                 pos = k, condition = containsSlash)
        all_letters_context = checkPreviousTokenCondition(tokens = tokenized_sentence, 
                                                                 pos = k, condition = allLetters)
        all_digits_context = checkPreviousTokenCondition(tokens = tokenized_sentence, 
                                                                 pos = k, condition = allDigits)
        contains_digit_context = checkPreviousTokenCondition(tokens = tokenized_sentence, 
                                                                 pos = k, condition = hasNumbers)
        contains_letters_context = checkPreviousTokenCondition(tokens = tokenized_sentence, 
                                                                 pos = k, condition = hasLetters)
        contains_uppercase_context = checkPreviousTokenCondition(tokens = tokenized_sentence, 
                                                                 pos = k, condition = hasUpperCase)
        contains_dash_context = checkPreviousTokenCondition(tokens = tokenized_sentence, 
                                                                 pos = k, condition = containsDash)
        '''
        
        unigrams = checkPreviousTokenCondition(tokens = tokenized_sentence, pos = k, condition = num_unigrams)
        
        bigrams = checkPreviousTokenCondition(tokens = tokenized_sentence, pos = k, condition = num_bigrams)
        
        trigrams = checkPreviousTokenCondition(tokens = tokenized_sentence, pos = k, condition = num_trigrams)

        '''
        feature_vector['all_uppercase_letters_context%d' % k] = all_uppercase_letters_context
        feature_vector['all_lowercase_letters_context%d' % k] = all_lowercase_letters_context
        feature_vector['initial_capital_letter_context%d' % k] = initial_capital_letter_context
        feature_vector['contains_slash_context%d' % k] = contains_slash_context
        feature_vector['all_letters_context%d' % k] = all_letters_context
        feature_vector['all_digits_context%d' % k] = all_digits_context
        feature_vector['contains_digit_context%d' % k] = contains_digit_context
        feature_vector['contains_letters_context%d' % k] = contains_letters_context
        feature_vector['contains_uppercase_context%d' % k] = contains_uppercase_context
        feature_vector['contains_dash_context%d' % k] = contains_dash_context
        '''
        
        feature_vector['unigrams%d' % k] = unigrams
        feature_vector['bigrams%d' % k] = bigrams
        feature_vector['trigrams%d' % k] = trigrams

    # Orthographic features taking profit of the checkPreviousTokenCondition. We want this only for k =0
    '''  
    feature_vector['all_uppercase_letters']= checkPreviousTokenCondition(tokens = tokenized_sentence, 
                                                                 pos = 0, condition = allCaps)
    feature_vector['all_lowercase_letters'] = checkPreviousTokenCondition(tokens = tokenized_sentence, 
                                                                 pos = 0, condition = allLowerCase)
    feature_vector['initial_capital_letter'] = checkPreviousTokenCondition(tokens = tokenized_sentence, 
                                                                 pos = 0, condition = hasInitialCapital)
    feature_vector['contains_slash'] = checkPreviousTokenCondition(tokens = tokenized_sentence, 
                                                                 pos = 0, condition = containsSlash)
    feature_vector['all_letters'] = checkPreviousTokenCondition(tokens = tokenized_sentence, 
                                                                 pos = 0, condition = allLetters)
    feature_vector['all_digits'] = checkPreviousTokenCondition(tokens = tokenized_sentence, 
                                                                 pos = 0, condition = allDigits)
    feature_vector['contains_digit'] = checkPreviousTokenCondition(tokens = tokenized_sentence, 
                                                                 pos = 0, condition = hasNumbers)
    feature_vector['contains_letters'] = checkPreviousTokenCondition(tokens = tokenized_sentence, 
                                                                 pos = 0, condition = hasLetters)
    feature_vector['contains_uppercase'] = checkPreviousTokenCondition(tokens = tokenized_sentence, 
                                                                 pos = 0, condition = hasUpperCase)
    feature_vector['contains_dash_contex'] = checkPreviousTokenCondition(tokens = tokenized_sentence, 
                                                                pos = 0, condition = containsDash)
    '''
    
    # Feature: Check if the token is present in the DrugBank database (previously parsed)

    is_token_in_DrugBank_db = []
    for token in tokenized_sentence:
        for drug in drugbank_db:
            if drug in token.lower():
                is_token_in_DrugBank_db.append(1)
                break
        else:
            is_token_in_DrugBank_db.append(0)
    
    feature_vector['is_token_in_DrugBank_db'] = is_token_in_DrugBank_db
    
    '''
    # Feature: Map all tokens to the Aa1- format
    
    feature_vector['Aa1-'] = tokenToAaFormat(tokenized_sentence)
    '''
    
    # Feature: Keep the most rare (in terms of frequency) words in the alphabet
    
    return feature_vector

def frequencyTokens(tokens):
    fdist = nltk.FreqDist(list(set(tokens))) # list of tuples with the following format (token,freq)
    frequencies = [fdist[token] for token in tokens]
    return frequencies

    
def tokenToAaFormat(tokens):
    # the idea is to aggrupate words (a kind of clustering) with the following mapping criteria:
    # any uppercase letter ---- 'A'
    # any lower case letter --- 'a'
    # any sign ---- '-'
    # any number ---- '1'
    # TODO: Add tests with examples
    new_tokens = []
    for token in tokens:
        word = ''
        for char in token:
            if char.isupper():
                word = word+'A'
            elif char.islower():
                word = word+'a'
            elif char.isdigit():
                word = word+'1'
            elif char in '*&$-/%·#[]()\!_.:, ;=~+':
                word = word+'-'
            else: warnings.warn('Character "' + char + '"" could not be mapped to any of the options')

        new_tokens.append(''.join(set(sorted(word)))) # I just compress any substring with the same letter and order it!
    return new_tokens


            
if __name__ == '__main__':
    #print(BOTagger('Give me TNF antioxidants together with Exter diodorant sodium, please', ['TNF antioxidants', 'Exter diodorant sodium']))
    import doctest
    doctest.testmod()
    