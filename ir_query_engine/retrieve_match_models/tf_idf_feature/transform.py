from gensim import corpora, models, similarities
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer

__author__ = 'Deyang'



tokenizer = RegexpTokenizer(r'\w+')
p_stemmer = PorterStemmer()

# create sample documents
doc_a = "Brocolli is good to eat. My brother likes to eat good brocolli, but not my mother."
doc_b = "My mother spends a lot of time driving my brother around to baseball practice."
doc_c = "Some health experts suggest that driving may cause increased tension and blood pressure."
doc_d = "I often feel pressure to perform well at school, but my mother never seems to drive my brother to do better."
doc_e = "Health professionals say that brocolli is good for your health."

# compile sample documents into a list
doc_set = [doc_a, doc_b, doc_c, doc_d, doc_e]


def pre_process_doc_tf_idf(raw_doc):
    """
    Pre-processing fortf-idf does not
    1) rm stop words
    2) rm low-fre words

    :param raw_doc:
    :return:
    """
    # clean and tokenize document string
    tokens = tokenizer.tokenize(raw_doc.lower())
    # stem tokens
    stemmed_tokens = [p_stemmer.stem(t) for t in tokens]
    return stemmed_tokens


def docs_to_corpus_tf_idf(doc_set):
    # list for tokenized documents in loop
    texts = []

    # loop through document list
    for i in doc_set:
        # add tokens to list
        texts.append(pre_process_doc_tf_idf(i))

    # turn our tokenized documents into a id <-> term dictionary
    dictionary = corpora.Dictionary(texts)
    # convert tokenized documents into a document-term matrix
    corpus = [dictionary.doc2bow(text) for text in texts]
    return dictionary, corpus


dictionary, corpus = docs_to_corpus_tf_idf(doc_set)
print dictionary
print corpus
tfidf = models.TfidfModel(corpus)



# query part
doc_f = "brocolli is good for healthy."

vec = tfidf[dictionary.doc2bow(pre_process_doc_tf_idf(doc_f))]

index = similarities.SparseMatrixSimilarity(tfidf[corpus], num_features=47)


sims = index[vec]
print(list(enumerate(sims)))
