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

# Logistic Regression

from sklearn import metrics
from sklearn.linear_model import LogisticRegression

data = [
    [0.1, 0.2, 0.3, 0.4],
    [0.9, 0.8, 0.4, 0.5],
    [0.4, 0.2, 0.1, 0.0],
    [-0.3, -2.1, -3.2, -3.4],
]
label = [1, 1, 0, 0]
# fit a logistic regression model to the data
model = LogisticRegression()
model.fit(data, label)

# # make predictions
test = [0.15, 0.25, 0.35, 0.5]
expected = 1
predicted = model.predict(test)
print predicted
# # summarize the fit of the model
print(metrics.classification_report([expected], predicted))
print(metrics.confusion_matrix([expected], predicted))


