import nltk,warnings,re
import pandas as pd
from nltk.tag import StanfordPOSTagger


# Define precision and recall function for model evaluation
def compute_recall(pred_ent,true_ent):
    if len(pred_ent) == 0 or len(true_ent) == 0: return 0
    else: return round(len([word for word in pred_ent if word in true_ent])/len(true_ent),2)*100


def compute_precision(pred_ent,true_ent):
    if len(pred_ent) == 0 or len(true_ent) == 0: return 0
    else: return round(len([word for word in pred_ent if word in true_ent])/len(pred_ent),2)*100     
    

# Define some functions that will be used in order to create the features
def hasNumbers(string):
    return any(char.isdigit() for char in string)

def hasLetters(string):
    return any(char.isalpha() for char in string)

def hasUpperCase(string):
    return any(char.isupper() for char in string)

def allUpperCase(string):
    return(string.isupper())

def allLowerCase(string):
    return(string.islower())

def hasInitialCapital(string):
    return(string[0].isupper())

def containsSlash(string):
    return('/' in string)

def allLetters(string):
    return(string.isalpha())

def allDigits(string):
    return(string.isdigit())

def containsDash(string):
    return('-' in string)

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
    

def bioTagger(text, drugs):
    ''' (str, list of str) -> list of tuples
    
    Description: 
    Given a sentence 'text' and a set of drugs 'drugs', this function returns a list of str that
    contains a tag for each of the tokens in text. The tags can be either 'B', 'I' or 'O'. 'B' means
    the token is the first part of a drug entity, 'I' means the token is the continuation of a drug entity,
    and 'O' means that the token does not belong to a drug entity.
    
    Examples/Tests:
    
    >>> bioTagger('Ibuprofeno is great!', ['Ibuprofeno'])
    [('Ibuprofeno', 'B'), ('is', 'O'), ('great', 'O'), ('!', 'O')]
    >>> bioTagger('I would like to buy calcium-rich milk', ['calcium'])
    [('I', 'O'), ('would', 'O'), ('like', 'O'), ('to', 'O'), ('buy', 'O'), ('calcium-rich', 'B'), ('milk', 'O')]
    >>> bioTagger('Give me TNF antioxidants together with sodium, please', ['TNF antioxidants', 'sodium'])
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


def bioTagsToEntities(tokens, bio_tags):
    '''
    Description:
    Given a list of tokens, 'tokens' and a list of tags for each token, 'bio_tags', this function returns a list of all the entities
    detected that had a 'B' or a 'I' associated. This function will be used in order to retrieve the entities predicted by the classification
    model and compare them with the real ones. 

    Examples/Tests:
    >>> bioTagsToEntities(tokens = ['START', 'Ibuprofeno', 'is', 'good', '.'], bio_tags = ['O', 'B', 'O', 'O', 'O'])
    ['Ibuprofeno']
    >>> bioTagsToEntities(tokens = ['START', 'Food', 'is', 'good', '.'], bio_tags = [])
    []
    >>> bioTagsToEntities(tokens = ['START', 'TNF', 'Receptors', 'are', 'good', '.', 'STOP'], bio_tags = ['O', 'B', 'I', 'O', 'O', 'O', 'O'])
    ['TNF Receptors']
    >>> bioTagsToEntities(tokens = ['START', 'TNF', 'Receptors', 'and', 'Fluimicil', '400mg', 'are', 'good', '.', 'STOP'], bio_tags = ['O', 'B', 'I', 'O', 'B', 'I', 'O', 'O', 'O', 'O'])
    ['TNF Receptors', 'Fluimicil 400mg']
    >>> bioTagsToEntities(tokens = ['START', 'TNF', 'Receptors', 'and', 'Fluimicil', '400mg', 'are', 'good', '.', 'STOP'], bio_tags = ['O', 'B', 'I', 'B', 'I', 'I', 'O', 'O', 'O', 'O'])
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

def createFeatureVector(sentence, drugbank_db,st):
    ''' (string, list) -> dict
    Description:
    
    Examples/Tests:
    
    '''
    tokenized_sentence = nltk.word_tokenize(sentence)

    # Feature: Initialise the feature_vector data frame, in which we will create the features of each token
    feature_vector = pd.DataFrame()
    
    # Feature: Length of the token
    feature_vector['token_length'] = [len(token) for token in tokenized_sentence]
    
    # Feature: Prefixes and Suffixes

    prefix_feature = []
    suffix_feature = []

    prefixes = r'^meth|^eth|^prop|^but|^pent|^hex|^hept|^oct|^non|^dec'
    suffixes = r'ane$|ene$|yne$|ol$|al$|amine$|cid$|ium$|ether$|ate$|one$'

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

    
    # Feature: POS of the token with Stanford POStagger
    
    '''
    tuples = st.tag(tokenized_sentence)
    pos = [word[1] for word in tuples]
    pos_tags=pd.DataFrame({'pos_tags':pos})
    one_hot = pd.get_dummies(pos_tags['pos_tags'])
    
    
    # one hot coding will create columns for those tags seen according to the token set in question. In order to avoid
    # NaN's when doing the join of the two data frames, we will have to create columns of 0's according to the other 
    # possible pos tags found with nltk
    
    nltk_pos_tags = ["''",'$','(',')',',','--','.',':','CC','CD','DT','EX','FW','IN','JJ','JJR','JJS','LS','MD','NN','NNP','NNPS','NNS','PDT','POS','PRP','PRP$','RB','RBR','RBS','RP','SYM','TO','UH','VB','VBD','VBG','VBN','VBP','VBZ','WDT','WP','WP$','WRB',"``"]
    for name in nltk_pos_tags:
        if name not in one_hot.columns.values:
            one_hot[name]=[0]*len(one_hot[one_hot.columns.values[0]])
    
    # joining both data frames
    feature_vector = feature_vector.join(one_hot)
    '''
    
    # Feature: Binary token type features
        # contains_hyphen, all_lowercase_letters, 
        # contains_slash, all_letters, contains_period, all_digits, contains_uppercase,
        # contains_digit, contains_letters
    
    all_uppercase_letters = [1 if token.isupper() else 0 for token in tokenized_sentence]
    all_lowercase_letters = [1 if token.islower() else 0 for token in tokenized_sentence]
    initial_capital_letter = [1 if token[0].isupper() else 0 for token in tokenized_sentence]
    contains_slash = [1 if '/' in token else 0 for token in tokenized_sentence]
    all_letters = [1 if token.isalpha() else 0 for token in tokenized_sentence]
    all_digits = [1 if token.isdigit() else 0 for token in tokenized_sentence]
    contains_digit = [1 if hasNumbers(token) else 0 for token in tokenized_sentence]
    contains_letters = [1 if hasLetters(token) else 0 for token in tokenized_sentence]
    contains_uppercase = [1 if hasUpperCase(token) else 0 for token in tokenized_sentence]
    contains_dash = [1 if '_' in token else 0 for token in tokenized_sentence]

    
    feature_vector['all_uppercase_letters']=all_uppercase_letters
    feature_vector['all_lowercase_letters']=all_lowercase_letters
    feature_vector['initial_capital_letter']=initial_capital_letter
    feature_vector['contains_slash']=contains_slash
    feature_vector['all_letters']=all_uppercase_letters
    feature_vector['all_digits']=all_digits
    feature_vector['contains_digit']=contains_digit
    feature_vector['contains_letters']=contains_letters
    feature_vector['contains_uppercase']=contains_uppercase  
    feature_vector['contains_dash']=contains_dash  
    
    
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
    
    for k in [-2, -1, 1, 2]:
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
    return feature_vector

    
def tokenToAaFormat(tokens):
    # the idea is to aggrupate words (a kind of clustering) with the following mapping criteria:
    # any uppercase letter ---- 'A'
    # any lower case letter --- 'a'
    # any sign ---- '-'
    # any number ---- '1'
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
            elif char in '*&$-/%·#[]()\!_.,':
                word = word+'-'
            else: warnings.warn('This character could not be mapped to any of the options')


        new_tokens.append(word)
    return new_tokens


            
if __name__ == '__main__':
    import doctest
    doctest.testmod()
    