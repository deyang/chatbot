from gensim import corpora, models, similarities
from ir_query_engine.retrieve_match_models.tf_idf_feature.transform import pre_process_doc_tf_idf, docs_to_corpus_tf_idf


__author__ = 'Deyang'



# create sample documents
doc_a = "Brocolli is good to eat. My brother likes to eat good brocolli, but not my mother."
doc_b = "My mother spends a lot of time driving my brother around to baseball practice."
doc_c = "Some health experts suggest that driving may cause increased tension and blood pressure."
doc_d = "I often feel pressure to perform well at school, but my mother never seems to drive my brother to do better."
doc_e = "Health professionals say that brocolli is good for your health."

# compile sample documents into a list
doc_set = [doc_a, doc_b, doc_c, doc_d, doc_e]


dictionary, corpus = docs_to_corpus_tf_idf(doc_set)
tfidf = models.TfidfModel(corpus)

# query part
doc_f = "brocolli is good for healthy."



# tf BOW
tf_vec = dictionary.doc2bow(pre_process_doc_tf_idf(doc_f))
# idf from model
idf_vec = vector = [(termid, tfidf.idfs.get(termid))
                    for termid, tf in tf_vec if tfidf.idfs.get(termid, 0.0) != 0.0]


# learn LR model