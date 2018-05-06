import nltk

def bio_tagger(text, drugs):
    ''' (str, list of str) -> list of tuples
    
    Description: 
    Given a sentence 'text' and a set of drugs 'drugs', this function returns a list of str that
    contains a tag for each of the tokens in text. The tags can be either 'B', 'I' or 'O'. 'B' means
    the token is the first part of a drug entity, 'I' means the token is the continuation of a drug entity,
    and 'O' means that the token does not belong to a drug entity.
    
    Examples/Tests:
    
    >>> bio_tagger('Ibuprofeno is great!', ['Ibuprofeno'])
    [('Ibuprofeno', 'B'), ('is', 'O'), ('great', 'O'), ('!', 'O')]
    >>> bio_tagger('I would like to buy calcium-rich milk', ['calcium'])
    [('I', 'O'), ('would', 'O'), ('like', 'O'), ('to', 'O'), ('buy', 'O'), ('calcium-rich', 'B'), ('milk', 'O')]
    >>> bio_tagger('Give me TNF antioxidants together with sodium, please', ['TNF antioxidants', 'sodium'])
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


            
if __name__ == '__main__':
    import doctest
    doctest.testmod()
    