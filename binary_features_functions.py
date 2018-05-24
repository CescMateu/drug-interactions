# Define some functions that will be used in order to create ortographic features
def hasNumbers(string):
    return int(any(char.isdigit() for char in string))

def hasLetters(string):
    return int(any(char.isalpha() for char in string))

def hasUpperCase(string):
    return int(any(char.isupper() for char in string))

def allUpperCase(string):
    return(int(string.isupper()))

def allLowerCase(string):
    return(int(string.islower()))

def hasInitialCapital(string):
    return(int(string[0].isupper()))

def containsSlash(string):
    return(int('/' in string))

def allLetters(string):
    return(int(string.isalpha()))

def allDigits(string):
    return(int(string.isdigit()))

def containsDash(string):
    return(int('-' in string))

def countTokensEntity(entity):
    entity_tokenized = nltk.word_tokenize(entity)
    return(len(entity_tokenized))