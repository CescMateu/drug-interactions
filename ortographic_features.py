# Import basic packages
import nltk
import os
import pandas as pd

# Import self-defined functions
from drug_functions import *
from binary_features_functions import *


def getPrefix(entity, prefixes_re):
	'''

	Examples:
	>>> prefixes = r'^alk|^meth|^eth|^prop|^but|^pent|^hex|^hept|^oct|^non|^dec|^undec|^dodec|^eifcos|^di|^tri|^tetra|^penta|^hexa|^hepta'
	>>> getPrefix('ethanol', prefixes)
	'eth'
	>>> prefixes = r'^alk|^meth|^eth|^prop|^but|^pent|^hex|^hept|^oct|^non|^dec|^undec|^dodec|^eifcos|^di|^tri|^tetra|^penta|^hexa|^hepta'
	>>> getPrefix('ibuprofeno', prefixes)
	'none'
	'''

	#prog = re.compile(pattern)
	result = re.search(prefixes_re, entity)

	if result is None:
		return('none')
	else:
		return(result.group())

def getSuffix(entity, suffixes_re):
	'''

	Examples:
	>>> suffixes = suffixes = r'ane$|ene$|yne$|yl$|ol$|al$|oic$|one$|ate$|amine$|amide$'
	>>> getPrefix('ethanol', suffixes)
	'ol'
	>>> suffixes = r'ane$|ene$|yne$|yl$|ol$|al$|oic$|one$|ate$|amine$|amide$'
	>>> getPrefix('ibuprofeno', suffixes)
	'none'
	'''

	#prog = re.compile(pattern)
	result = re.search(suffixes_re, entity)

	if result is None:
		return('none')
	else:
		return(result.group())



if __name__ == '__main__':
    import doctest
    doctest.testmod()

