import nltk
import re

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

# Orthographic features from paper
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