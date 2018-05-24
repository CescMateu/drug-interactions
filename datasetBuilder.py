from lxml import etree # XML file parsing
from os import listdir 

def createTrainSet(train_dirs_whereto_parse):
    # Initialise the different lists with the data
    entities=[]
    texts=[]
    train_texts_entities = []

    # Iterate over all the different .xml files located in the specified directories
    for directory in train_dirs_whereto_parse:
        
        # Get the names of all the files in the directory and create a 'xml.root' object for
        # each xml file
        roots = [etree.parse(directory+'/'+a).getroot() for a in listdir(directory) if a.endswith('.xml')]
        
        # Iterate over all the different 'xml.root' objects to extract the needed information
        for root in roots:
            for sentence in root.findall('sentence'):
                for entity in sentence.findall('entity'):
                    entities = entities+[entity.get('text')]
                # we do not add to the train set those sentences with no entities
                if entities:
                    #train_texts_entities = train_texts_entities + [('START '+sentence.get('text')+' STOP', entities)]
                    train_texts_entities = train_texts_entities + [(sentence.get('text'), entities)]
                    entities =[]

    # train_texts_entities is a list of tuples. Each one is comprised of the sentence and the drugs in there
    # Example: 
    # [('I love Ibuprofeno and Frenadol', ['Ibuprofeno', 'Frenadol']), ('Give me a Fluimucil', ['Fluimucil'])]
    return train_texts_entities

def createTestSet(test_dirs_whereto_parse):
    ## TESTING DATA

    # Same process as with the training data
    # In the testing data, for each sentance we have two related files:
    # - A file with a sentence to be parsed, in which we may encounter drug names (ending with 'text.txt')
    # - A file with the drug entities recognised in the sentence (ending with 'entities.txt')

    test_texts = []
    test_entities = []

    for directory in test_dirs_whereto_parse:
        
        # Si no poso el sorted, em llegeix els files amb un ordre aleatori.
        # Amb el sorted m'asseguro que els corresponents files text.txt i entities.txt estan en la mateixa posicio
        
        # Read the pairs of files in alphabetical order
        text_file_names = sorted([directory + '/' + file for file in listdir(directory) if file.endswith('text.txt')])
        entities_file_names = sorted([directory + '/' + file for file in listdir(directory) if file.endswith('entities.txt')])
        
        for file in text_file_names:
            file = open(file,'r')
            #test_texts = test_texts + ['START '+file.read()[:-1]+' STOP'] # each file.read() string ends with a \n I do not want
            test_texts = test_texts + [file.read()[:-1]]
        for file in entities_file_names:
            read_entities = []
            with open(file,'r') as f:
                for line in f:
                    read_entities = read_entities+[' '.join(line.split()[0:-1])] # separo en words, el.limino la ultima i torno a unir
                    
            test_entities.append(read_entities)


    test_texts_entities=list(zip(test_texts,test_entities))


    # test_texts_entities is a list of tuples. Each one is comprised of the sentence and the drugs in there.
    return test_texts_entities

