from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models, similarities
import os

__author__ = 'Deyang'

tokenizer = RegexpTokenizer(r'\w+')

# create English stop words list
en_stop = get_stop_words('en')

# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()

# create sample documents
doc_a = "Brocolli is good to eat. My brother likes to eat good brocolli, but not my mother."
doc_b = "My mother spends a lot of time driving my brother around to baseball practice."
doc_c = "Some health experts suggest that driving may cause increased tension and blood pressure."
doc_d = "I often feel pressure to perform well at school, but my mother never seems to drive my brother to do better."
doc_e = "Health professionals say that brocolli is good for your health."

# compile sample documents into a list
doc_set = [doc_a, doc_b, doc_c, doc_d, doc_e]


def pre_process_doc_lda(raw_doc):
    # clean and tokenize document string
    tokens = tokenizer.tokenize(raw_doc.lower())

    # remove stop words from tokens
    stopped_tokens = [t for t in tokens if t not in en_stop]

    # stem tokens
    stemmed_tokens = [p_stemmer.stem(t) for t in stopped_tokens]
    return stemmed_tokens


def get_model_file_path():
    filename = 'lda.md'
    dirpath = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(dirpath, filename)


def get_dictionary_file_path():
    filename = 'lda.dict'
    dirpath = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(dirpath, filename)


def docs_to_corpus_lda(doc_set):
    # list for tokenized documents in loop
    texts = []

    # loop through document list
    for i in doc_set:
        # add tokens to list
        texts.append(pre_process_doc_lda(i))

    # turn our tokenized documents into a id <-> term dictionary
    dictionary = corpora.Dictionary(texts)
    # convert tokenized documents into a document-term matrix
    corpus = [dictionary.doc2bow(text) for text in texts]
    return dictionary, corpus




dictionary, corpus = docs_to_corpus_lda(doc_set)

# generate LDA model
ldamodel = models.ldamodel.LdaModel(corpus, num_topics=3, id2word=dictionary, passes=20)



predict = ldamodel[dictionary.doc2bow(pre_process_doc_lda(doc_a))]
print predict


ldamodel.save(get_model_file_path())


####### test save and load
dictionary.save_as_text(get_dictionary_file_path())



# the real query part
new_dict = corpora.Dictionary.load_from_text(get_dictionary_file_path())
new_model = models.ldamodel.LdaModel.load(get_model_file_path())

predict = new_model[new_dict.doc2bow(pre_process_doc_lda(doc_a))]
print predict


index = similarities.MatrixSimilarity(new_model[corpus], num_features=3)

print index[predict]