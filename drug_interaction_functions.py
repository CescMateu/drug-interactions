import nltk

def countTokensBetweenEntities(sentence, ent1, ent2):
	'''

	Description:
	Returns an integer indicating the number of tokens between entity1 and entity2, independently from the real order in the sentence.
	Entity1 and entity2 are also counted in the integer returned. If the entity is not in the sentence, an exception is raised.

	Examples:
	>>> countTokensBetweenEntities('My name is Cesc Mateu', 'My', 'Cesc')
	4
	>>> countTokensBetweenEntities('My name is Cesc Mateu', 'Cesc', 'My')
	4
	>>> countTokensBetweenEntities('I am listening to music', 'music', 'music')
	1
	>>> countTokensBetweenEntities('I am listening to music', 'to', 'music')
	2
	'''
	sentence_token = nltk.word_tokenize(sentence)
	return(abs(sentence_token.index(ent1)-sentence_token.index(ent2)) + 1)

def countEntitiesBetweenEntities(sentence, ent1, ent2, entities_list):
    '''
	(str, str, str, list of str) -> int
    Description:
    Returns an integer indicating the number of entities between ent1 and ent2. A list of entities is provided in entities_list which includes
    ent1 and ent2. 

    Examples:
    >>> countEntitiesBetweenEntities('Right now I am trying to work', 'now', 'trying', ['work', 'now', 'trying', 'am'])
    1
    >>> countEntitiesBetweenEntities('Oops, seems that I took my Ibuprofeno too late', 'Ibuprofeno', 'seems', ['seems', 'Ibuprofeno', 'took', 'I', 'Oops'])
    2
    >>> countEntitiesBetweenEntities('Oops, seems that I took my Ibuprofeno too late', 'Ibuprofeno', 'seems', ['seems', 'Ibuprofeno', 'late', 'Oops'])
    0
    >>> countEntitiesBetweenEntities('Oops, seems that I took my Ibuprofeno too late', 'Ibuprofeno', 'seems', ['seems', 'Ibuprofeno'])
    0
    '''

    # Debugging
    #import pdb; pdb.set_trace()
    
    # Eliminate the reference entities from the list of entities
    entities_list.remove(ent1)
    entities_list.remove(ent2)

    # If the length of the entities_list is 0, return a 0 (Only two entities were provided)
    if len(entities_list) == 0:
        return(0)

    # Tokenize the sentence
    sentence_tokenized = nltk.word_tokenize(sentence)

    # Get the range of the sentence in which to iterate
    if sentence_tokenized.index(ent1) <	sentence_tokenized.index(ent2):
        min_idx = sentence_tokenized.index(ent1)
        max_idx = sentence_tokenized.index(ent2)
    else:
        min_idx = sentence_tokenized.index(ent2)
        max_idx = sentence_tokenized.index(ent1)

    # Iterate over the sentence between ent1 and ent2 looking for other entities
    counter_entities = 0
    for idx in range(min_idx, max_idx):
        if sentence_tokenized[idx] in entities_list:
            entities_list.remove(sentence_tokenized[idx])
            counter_entities += 1

    return(counter_entities)

def countModalVerbsBetweenEntities(sentence, ent1, ent2):
	'''
	Description:
	Returns an integer indicating the number of modal verbs in the sentence between ent1 and ent2

	>>> countModalVerbsBetweenEntities('I think you should eat your ice-cream', 'you', 'ice-cream')
	1
	>>> countModalVerbsBetweenEntities('I think you should eat your ice-cream', 'your', 'ice-cream')
	0
	'''
	# Initializate a list with the most common modal verbs
	modal_verbs = ['should', 'must', 'can', 'could', 'may', 'might', 'will', 'would', 'shall']

	# Tokenize the sentence
	sentence_tokenized = nltk.word_tokenize(sentence)

	# Get the range of the sentence in which to iterate
	if sentence_tokenized.index(ent1) <	sentence_tokenized.index(ent2):
	    min_idx = sentence_tokenized.index(ent1)
	    max_idx = sentence_tokenized.index(ent2)
	else:
	    min_idx = sentence_tokenized.index(ent2)
	    max_idx = sentence_tokenized.index(ent1)

	# Iterate over the list to retrieve the result
	modal_verbs_count = 0
	for modal_verb in modal_verbs:
		if modal_verb in sentence_tokenized[min_idx:max_idx+1]:
			modal_verbs_count += 1

	return(modal_verbs_count) 


if __name__ == '__main__':
    import doctest
    doctest.testmod()
    