from __future__ import unicode_literals
import word2vec
import os

# Use this on production ec2 instance
filename = 'GoogleNews-vectors-negative300.bin'
# Use this on mac dev env
filename = 'vectors.bin'
dirpath = os.path.dirname(os.path.abspath(__file__))
filepath = os.path.join(dirpath, filename)
print filepath
model = word2vec.load(filepath, encoding='ISO-8859-1')

print model['apples']
print model['dogs'].shape