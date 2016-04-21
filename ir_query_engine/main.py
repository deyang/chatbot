from optparse import OptionParser

from common import load_data
import retrieve_match_models.tf_idf_feature.tfidf_model as tfidf_model
import retrieve_match_models.lda_feature.lda_model as lda_train
import rank_match_models.topic_word_feature.topic_word_model as topic_train
from rank_match_models.word2vec_feature.word2vec_model import Word2VecModel
from ranker.ranking import Matcher, LinearRankModel
from query_engine import QueryEngine

__author__ = 'Deyang'


if __name__ == '__main__':

    parser = OptionParser()
    parser.add_option('-d', '--data_file', dest='data_file',
                      action='store',
                      default=None,
                      help='Input data file')
    parser.add_option('', '--load_tf_idf', dest='load_tf_idf',
                      action='store_true',
                      default=False,
                      help='Load TF-IDF model only')
    parser.add_option('', '--train_lda', dest='train_lda',
                      action='store_true',
                      default=False,
                      help='Train LDA model only')
    parser.add_option('', '--train_topic_words', dest='train_topic_words',
                      action='store_true',
                      default=False,
                      help='Train topic words model')
    parser.add_option('', '--train_rank_model', dest='train_rank_model',
                      action='store_true',
                      default=False,
                      help='Train rank model')
    parser.add_option('-q', '--query', dest='query',
                      action='store_true',
                      default=False,
                      help='Start a QueryEngine')
    parser.add_option('', '--num_topics', dest='num_topics',
                      action='store',
                      default=None,
                      help='Number of topics in LDA model.')
    parser.add_option('-r', '--regen', dest='regen',
                      action='store_true',
                      default=False,
                      help='Force to re gen/train models')

    (options, args) = parser.parse_args()

    if options.data_file:
        data_store = load_data(options.data_file)

    if options.load_tf_idf:
        tf_idf_model_struct = tfidf_model.TfIdfModelStruct.get_model(data_store=data_store, regen=options.regen)
        raw_doc = "Mark Zuckerberg established Facebook"
        results = tf_idf_model_struct.query_questions(raw_doc=raw_doc)
        results = data_store.translate_question_query_results(results)
        print results
        print data_store.doc_set[results[0][0]]

        results = tf_idf_model_struct.query_answers(raw_doc=raw_doc)
        results = data_store.translate_answer_query_results(results)
        print results
        print data_store.doc_set[results[0][0]]

    if options.train_lda:
        lda_model_struct = \
            lda_train.LdaModelStruct.get_model(data_store=data_store, regen=options.regen, num_topics=int(options.num_topics))
        results = lda_model_struct.query(raw_doc="Mark Zuckerberg is a great founder")
        print results
        print data_store.doc_set[results[0][0]]
        print data_store.doc_set[results[1][0]]
        print data_store.doc_set[results[2][0]]

    if options.train_topic_words:
        tf_idf_model_struct = tfidf_model.TfIdfModelStruct.get_model(data_store=data_store)
        topic_word_model_struct = \
                topic_train.TopicWordModelStruct.get_model(
                    tfidf_model_struct=tf_idf_model_struct,
                    data_store=data_store,
                    regen=options.regen)

        query_doc = "What is investment strategy"
        compare_docs = data_store.doc_set
        results = topic_word_model_struct.get_similarities(query_doc, compare_docs)
        print compare_docs[results[0][0]]
        print compare_docs[results[1][0]]
        print compare_docs[results[2][0]]
        print compare_docs[results[3][0]]

    if options.train_rank_model:
        tfidf_model_struct = tfidf_model.TfIdfModelStruct.get_model()
        lda_model_struct = lda_train.LdaModelStruct.get_model(num_topics=options.num_topics)
        topic_word_model_struct = topic_train.TopicWordModelStruct.get_model(tfidf_model_struct=tfidf_model_struct)
        word2vec_model = Word2VecModel()
        matcher = Matcher(
            tfidf_model_struct,
            lda_model_struct,
            topic_word_model_struct,
            word2vec_model
        )

        rank_model = LinearRankModel(matcher=matcher, rank_data=data_store.rank_data)
        rank_model.write_training_data()

    if options.query:
        query_engine = QueryEngine(data_store, options.num_topics)
        # out_msg = query_engine.execute_query("facebook")
        while True:
            in_msg = raw_input()
            out_msg = query_engine.execute_query(in_msg)
            print "Response> %s" % out_msg
