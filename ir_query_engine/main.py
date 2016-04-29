from optparse import OptionParser

from common import load_data_store
import retrieve_match_models.tf_idf_feature.tfidf_model as tfidf_model
import retrieve_match_models.lda_feature.lda_model as lda_train
import rank_match_models.topic_word_feature.topic_word_model as topic_train
from rank_match_models.word2vec_feature.word2vec_model import Word2VecModel
import rank_match_models.topic_word_lookup_feature.topic_word_lookup_model as topic_word_lookup
from ranker.ranking import Matcher, RankTrainingDataGenerator, RankModelTrainer
from query_engine import QueryEngine
from utils.util import StopWatch
from ir_query_engine import engine_logger
from common import tokenizer, stop_words


__author__ = 'Deyang'


parser = OptionParser()
parser.add_option('-d', '--data_file', dest='data_file',
                  action='store',
                  default=None,
                  help='Input data file')
parser.add_option('', '--load_tfidf', dest='load_tfidf',
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
parser.add_option('', '--collect_topic_words', dest='collect_topic_words',
                  action='store_true',
                  default=False,
                  help='Lookup topic words model')
parser.add_option('', '--train_rank_model', dest='train_rank_model',
                  action='store_true',
                  default=False,
                  help='Train rank model')
parser.add_option('', '--kernel_type', dest='kernel_type',
                  action='store',
                  default='linear',
                  help='SVM kernel type: linear or rbf. Default: linear')
parser.add_option('', '--gamma', dest='gamma',
                  action='store',
                  default=0.5,
                  help='Gamma in rbf kernel function. Default: 0.5')
parser.add_option('-g', '--generate_rank_training_data', dest='generate_rank_training_data',
                  action='store_true',
                  default=False,
                  help='Generate rank training data to train rank model')
parser.add_option('-q', '--query', dest='query',
                  action='store_true',
                  default=False,
                  help='Start a QueryEngine')
parser.add_option('', '--write_rank_data', dest='write_rank_data',
                  action='store_true',
                  default=False,
                  help='Write the rank data sequentially')
parser.add_option('', '--pair', dest='pair',
                  action='store',
                  default=None,
                  help='Data split pair')
parser.add_option('', '--num_topics', dest='num_topics',
                  action='store',
                  default=None,
                  help='Number of topics in LDA model.')
parser.add_option('', '--num_splits', dest='num_splits',
                  action='store',
                  default=None,
                  help='Number of splits of rank training data.')
parser.add_option('-c', '--count_words', dest='count_words',
                  action='store_true',
                  default=False,
                  help='Count and print the most frequent words after rmed stop words')
parser.add_option('-r', '--regen', dest='regen',
                  action='store_true',
                  default=False,
                  help='Force to re gen/train models')

if __name__ == '__main__':
    (options, args) = parser.parse_args()

    if options.data_file:
        data_store = load_data_store(options.data_file)

    if options.count_words:
        counter = dict()
        for doc in data_store.doc_set:
            tokens = tokenizer.tokenize(doc.lower())

            for token in tokens:
                if token in stop_words:
                    continue
                if token not in counter:
                    counter[token] = 0
                counter[token] += 1

        items = list(counter.iteritems())
        items.sort(key=lambda t: t[1], reverse=True)
        print items[:50]

    if options.load_tfidf:
        tf_idf_model_struct = tfidf_model.TfIdfModelStruct.get_model(data_store=data_store, regen=options.regen)
        raw_doc = "Steven"
        results = tf_idf_model_struct.query_questions(raw_doc=raw_doc)
        results = data_store.translate_question_query_results(results)
        print results
        print data_store.doc_set[results[0][0]]

    if options.train_lda:
        lda_model_struct = \
            lda_train.LdaModelStruct.get_model(data_store=data_store, regen=options.regen, num_topics=int(options.num_topics))
        query_doc = "Mark Zuckerberg is a great founder"
        results = lda_model_struct.query(raw_doc=query_doc)
        print results
        print data_store.doc_set[results[0][0]]
        print data_store.doc_set[results[1][0]]
        print data_store.doc_set[results[2][0]]

    if options.collect_topic_words:
        topic_model_struct = \
            topic_word_lookup.TopicWordLookupModelStruct.get_model(
                data_store=data_store,
                regen=options.regen
            )

        query_doc = "What is steven"
        results = topic_model_struct.query(raw_doc=query_doc)
        print results[0:10]
        print data_store.doc_set[results[0][0]]

    if options.train_topic_words:
        tf_idf_model_struct = tfidf_model.TfIdfModelStruct.get_model(data_store=data_store)
        topic_word_model_struct = \
                topic_train.TopicWordModelStruct.get_model(
                    tfidf_model_struct=tf_idf_model_struct,
                    data_store=data_store,
                    regen=options.regen)

        query_doc = "What is invest"
        print query_doc
        compare_docs = data_store.doc_set
        results = topic_word_model_struct.get_similarities(query_doc, compare_docs)[0:10]
        results.sort(key=lambda t: t[1])
        print compare_docs[results[0][0]]
        print compare_docs[results[1][0]]
        print compare_docs[results[2][0]]
        print compare_docs[results[3][0]]

    if options.train_rank_model:
        c = len(data_store.rank_data)
        trainer = RankModelTrainer(options.kernel_type,
                                   len(data_store.rank_data),
                                   float(options.gamma))
        trainer.train_model()

    if options.generate_rank_training_data:
        total_num_data = len(data_store.rank_data)
        data = list(data_store.rank_data.iteritems())
        # sort by question to guarantee the order
        data.sort(key=lambda t: t[0])
        if options.write_rank_data:
            pair = options.pair.split(",")
            start_end = map(int, pair)
            sw = StopWatch()
            tfidf_model_struct = tfidf_model.TfIdfModelStruct.get_model()
            lda_model_struct = lda_train.LdaModelStruct.get_model()
            topic_word_lookup_model_struct = topic_word_lookup.TopicWordLookupModelStruct.get_model()
            word2vec_model = Word2VecModel()
            matcher = Matcher(
                tfidf_model_struct,
                lda_model_struct,
                topic_word_lookup_model_struct,
                word2vec_model
            )
            data_part = data[start_end[0]:start_end[1]]
            engine_logger.info("data pair: %s, length: %s" % (start_end, len(data_part)))
            rank_model = RankTrainingDataGenerator(
                matcher, data_part, query_id_offset=start_end[0])
            rank_model.write_training_data()
            print "total time to write rank data: %s seconds" % sw.stop()
        else:
            num_splits = int(options.num_splits)
            unit = total_num_data / num_splits
            pairs = []
            for i in range(num_splits):
                if i == num_splits - 1:
                    pairs.append((i*unit, total_num_data))
                else:
                    pairs.append((i*unit, (i+1)*unit))
            print pairs

    if options.query:
        query_engine = QueryEngine(data_store)
        # out_msg = query_engine.execute_query("facebook")
        while True:
            in_msg = raw_input()
            response = query_engine.execute_query(in_msg)
            print "Response> %s" % response.answer

