import numpy as np
import math
import zipfile
class WordReps():
	def __init__(self):
		self.vects=None
		self.vocab=None
#------------------------------------------------------------------
	def Read_Embeddings_zip_file(self,Embedding_file,dim):
		"""
        Read the word vectors where the first token in each line is the word.
        Input: word embedding filename (string), dimensionality of the embeddings (integer)
        
		"""
		self.dim=dim

		vects = {}
		vocab = []
		print ("Retrieving words embeddings...")
		zfile = zipfile.ZipFile(Embedding_file[0])
		for finfo in zfile.infolist():
			F = zfile.open(finfo)
			# read the vectors.
			line = F.readline()
			if len(line.split())==2:
				print ("Header Exists.")
				line=F.readline()
			while len(line) != 0:
				p = line.split()
				word = p[0].decode('UTF-8')
				v = np.zeros(self.dim, float)

				for i in range(0, self.dim):
					v[i] = float(p[i+1])
				vects[word] = v
				vocab.append(word)
				line = F.readline()
			print ("Number of words in the vocabulary is: ",len(vocab))
			F.close()
			self.vocab = vocab
			self.vects = vects
			break
#------------------------------------------------------------------
	def get_embedding(self,word):
		"""
	    If we can find the embedding for the word in vects, we will return it.
	    Otherwise, we will check if the lowercase version of word appears in vects
	    and if so we will return the embedding for the lowercase version of the word.
	    Otherwise we will return a zero vector.
		"""
		if word in self.vects:
			return self.vects[word]
		elif word.lower() in self.vects:
			return self.vects[word.lower()]
		else:
			return np.zeros(self.dim, dtype=float)
#------------------------------------------------------------