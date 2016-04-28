from ir_query_engine.query_engine import QueryEngine
from ir_query_engine.common import load_raw_data, load_data_store
from ir_query_engine.main import parser
from common import split_raw_data_k_fold
from ir_query_engine.retrieve_match_models.tf_idf_feature.tfidf_model import TfIdfModelStruct
from ir_query_engine.retrieve_match_models.lda_feature.lda_model import LdaModelStruct
from ir_query_engine import engine_logger
import json

__author__ = 'Deyang'

parser.add_option('', '--eval_tfidf', dest='eval_tfidf',
                  action='store_true',
                  default=False,
                  help='Evaluate TF-IDF model')
parser.add_option('', '--eval_lda', dest='eval_lda',
                  action='store_true',
                  default=False,
                  help='Evaluate LDA model')
parser.add_option('', '--num_folds', dest='num_folds',
                  action='store',
                  default=None,
                  help='Number of folds in k-fold cross validation')
parser.add_option('-w', '--write_output', dest='write_output',
                  action='store_true',
                  default=False,
                  help='Write experiments')


class SingleModelCrossValidationRunner(object):

    def __init__(self, data, num_folds, ModelClass, EvaluateClass, model_kargs):
        self.raw_data = data
        self.num_folds = num_folds
        self.ModelClass = ModelClass
        self.EvaluateClass = EvaluateClass
        self.model_kargs = model_kargs
        self.each_run_train_accuracy = []
        self.each_run_test_accuracy = []
        self.each_run_train_relevance = []
        self.each_run_test_relevance = []
        self.avg_train_accuracy = 0.0
        self.avg_test_accuracy = 0.0
        self.avg_train_relevance = 0.0
        self.avg_test_relevance = 0.0

    def cross_validate(self):
        iter_num = 0
        train_test_data_tuples = split_raw_data_k_fold(self.raw_data, self.num_folds)
        for train_data_store, train_data, test_data in train_test_data_tuples:
            engine_logger.info("Cross validation iter: %d" % iter_num)

            model = self.ModelClass.get_model(data_store=train_data_store,
                                              regen=True,
                                              **self.model_kargs)
            eval_model = self.EvaluateClass(
                train_data_store, train_data, test_data, model)
            eval_model.run_eval()

            train_accuracy, train_relevance, test_accuracy, test_relevance = eval_model.report_metrics()
            print train_accuracy, train_relevance, test_accuracy, test_relevance
            self.each_run_train_accuracy.append(train_accuracy)
            self.each_run_train_relevance.append(train_relevance)
            self.each_run_test_accuracy.append(test_accuracy)
            self.each_run_test_relevance.append(test_relevance)

            if iter_num == 0 and options.write_output:
                eval_model.write_output()

            iter_num += 1

        self.avg_train_accuracy = sum(self.each_run_train_accuracy) / float(self.num_folds)
        self.avg_train_relevance = sum(self.each_run_train_relevance) / float(self.num_folds)
        self.avg_test_accuracy = sum(self.each_run_test_accuracy) / float(self.num_folds)
        self.avg_test_relevance = sum(self.each_run_test_relevance) / float(self.num_folds)

    def report(self):
        return self.avg_train_accuracy, self.avg_train_relevance, self.avg_test_accuracy, self.avg_test_relevance


class EvaluateSingleModel(object):

    def __init__(self, train_data_store, train_data_set, test_data_set, model):
        """

        :param test_data_set: Test data set object
        :param model: Trained/Fit model
        :param query_func_name: the func attr name as string to query the most similar doc
        :param post_process_func: a function to translate the query results to a raw doc
        :return:
        """
        self.train_data_store = train_data_store
        self.train_data_set = train_data_set
        self.test_data_set = test_data_set
        self.model = model

    def run_eval(self):
        engine_logger.info("Evaluate on training data set")
        self._eval_on_data_set(self.train_data_set)
        engine_logger.info("Evaluate on test data set")
        self._eval_on_data_set(self.test_data_set)

    def _eval_on_data_set(self,  data_set):
        for idx in range(len(data_set.questions)):
            test_question = data_set.questions[idx]
            query_results = self.query(
                raw_doc=test_question)
            retrieved_result = self.post_process(query_results)
            data_set.retrieved_results.append(retrieved_result)
            if retrieved_result == data_set.top_answers[idx]:
                data_set.judgement_labels.append(1)
            else:
                data_set.judgement_labels.append(0)

            # compute relevance score
            sim = self.model.get_similarities(retrieved_result,
                                              [data_set.top_answers[idx]])[0][1]
            data_set.relevance_scores.append(sim)

        # compute metric
        data_set.accuracy = \
            sum(data_set.judgement_labels) / float(len(data_set.judgement_labels))
        data_set.avg_relevance_score = \
            sum(data_set.relevance_scores) / float(len(data_set.relevance_scores))

    def report_metrics(self):
        return self.train_data_set.accuracy, self.train_data_set.avg_relevance_score, \
               self.test_data_set.accuracy, self.test_data_set.avg_relevance_score

    def query(self, raw_doc):
        raise NotImplementedError

    def post_process(self, results):
        raise NotImplementedError

    def write_output(self):
        engine_logger.info("Writing output")
        with open('cv.log', 'w') as f:
            f.write("%f, %f\n" % (self.test_data_set.accuracy, self.test_data_set.avg_relevance_score))
            for idx in range(len(self.test_data_set.questions)):
                f.write("Question: %s\n" % self.test_data_set.questions[idx].encode('utf-8'))
                f.write("Correct answer: %s\n" % self.test_data_set.top_answers[idx].encode('utf-8'))
                f.write("Retrieved answer: %s\n" % self.test_data_set.retrieved_results[idx].encode('utf-8'))
                f.write("Label: %s, relevance score: %f\n" %
                        (self.test_data_set.judgement_labels[idx], self.test_data_set.relevance_scores[idx]))

            f.write(">>>>>>>> Training data\n")
            for idx in range(len(self.train_data_set.questions)):
                f.write("Question: %s\n" % self.train_data_set.questions[idx].encode('utf-8'))
                f.write("Answer: %s\n" % self.train_data_set.top_answers[idx].encode('utf-8'))


class EvaluateQuestionRetrieveSingleModel(EvaluateSingleModel):

    def query(self, raw_doc):
        return self.model.query_questions(raw_doc=raw_doc)

    def post_process(self, results):
        results = self.train_data_store.translate_question_query_results(results)
        qa_pair = self.train_data_store.get_docs_by_pair(self.train_data_store.qid_to_qa_pair[results[0][0]])
        return qa_pair[1]


class EvaluateDocRetrieveSingleModel(EvaluateSingleModel):

    def query(self, raw_doc):
        return self.model.query(raw_doc=raw_doc)

    def post_process(self, results):
        doc_id = results[0][0]
        if doc_id in self.train_data_store.question_set:
            qa_pair = self.train_data_store.qid_to_qa_pair[doc_id]
            return self.train_data_store.doc_set[qa_pair[1]]
        elif doc_id in self.train_data_store.answer_set:
            return self.train_data_store.doc_set[doc_id]
        return ""

def eval_score():
    data_store = load_data_store(options.data_file)
    query_engine = QueryEngine(data_store)

    exact_match_test_data = [
            # exact match, high score
            "who is Marc Andreessen",
            "tell me about Facebook",
            "where is Slack",
            "who founded 21",
            "show me the contact info of Andreessen Horowitz"
    ]
    relevant_match_test_data = [
            # relevant,
            "how to contact Andreessen Horowitz",
            "website of facebok",
            "who works for facebook",
            "funding of slack",
            "invest slack"
    ]
    missing_context_test_data = [
            # missing context
            "location",
            "who are the founders",
            "type of the business",
            "role",
            "funding",
            "picture",
    ]
    not_indexed_context_test_data = [
            # not indexed,
            "hello",
            "cool",
            "what is the color of sky",
            "who is ombama",
            "who is the president of USA",
            "what can you do"
    ]
    match_results = []
    for test_input in exact_match_test_data:
        response = query_engine.execute_query(test_input)
        match_results.append(
            (test_input, response.score)
        )
    relevant_results = []
    for test_input in relevant_match_test_data:
        response = query_engine.execute_query(test_input)
        relevant_results.append(
            (test_input, response.score)
        )
    missing_results = []
    for test_input in missing_context_test_data:
        response = query_engine.execute_query(test_input)
        missing_results.append(
            (test_input, response.score)
        )
    not_index_results = []
    for test_input in not_indexed_context_test_data:
        response = query_engine.execute_query(test_input)
        not_index_results.append(
            (test_input, response.score)
        )

    print "exact matches avg: %f" % (sum(map(lambda t: t[1], match_results)) / float(len(match_results)))
    print match_results

    print "relevant matches avg: %f" % (sum(map(lambda t: t[1], relevant_results)) / float(len(relevant_results)))
    print relevant_results

    print "missing matches avg: %f" % (sum(map(lambda t: t[1], missing_results)) / float(len(missing_results)))
    print missing_results

    print "not indexed matches avg: %f" % (sum(map(lambda t: t[1], not_index_results)) / float(len(not_index_results)))
    print not_index_results


if __name__ == '__main__':
    (options, args) = parser.parse_args()

    if options.data_file:
        raw_data = load_raw_data(options.data_file)

    if options.eval_tfidf:
        engine_logger.info("Cross validation on TFIDF model for %s folds" % options.num_folds)
        cv = SingleModelCrossValidationRunner(
            raw_data, int(options.num_folds), TfIdfModelStruct, EvaluateQuestionRetrieveSingleModel, {})
        cv.cross_validate()
        print cv.report()

    if options.eval_lda:
        engine_logger.info("Cross validation on LDA model for %s folds" % options.num_folds)
        cv = SingleModelCrossValidationRunner(
            raw_data,
            int(options.num_folds),
            LdaModelStruct,
            EvaluateDocRetrieveSingleModel,
            {'num_topics': int(options.num_topics)})
        cv.cross_validate()
        print cv.report()


