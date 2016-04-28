import unittest
import os

from mock import patch
from ir_query_engine.common import DataStore
from ir_query_engine.retrieve_match_models.tf_idf_feature.tfidf_model import TfIdfModelStruct
from gensim.models.tfidfmodel import df2idf
from gensim.matutils import cossim

__author__ = 'Deyang'


class TfIdfModelTestCase(unittest.TestCase):

    def setUp(self):
        self.data = [
            {
                "_question": "What do you like to eat?",
                "answer": "Brocolli is good to eat. My brother likes to eat good brocolli, but not my mother.",
            },
            {
                "_question": "Tell me about your mother.",
                "answer": "My mother spends a lot of time driving my brother around to baseball practice.",
            },
            {
                "_question": "Is driving safe?",
                "answer": "Some health experts suggest that driving may cause increased tension and blood pressure.",
            },
            {
                "_question": "Does your mother love you?",
                "answer": "I often feel pressure to perform well at school, but my mother never seems to drive my brother to do better.",
            },
            {
                "_question": "Want some brocolli?",
                "answer": "Health professionals say that brocolli is good for your health.",
            }
        ]
        self.data_store = DataStore(self.data)

        dir_path = os.path.dirname(os.path.abspath(__file__))
        self.test_md_file_path = os.path.join(dir_path, 'test_tfidf.md')
        self.test_dict_file_path = os.path.join(dir_path, 'test_tfidf.dict')
        self.test_q_simmx_file_path = os.path.join(dir_path, 'test_tfidf_q.simmx')
        self.test_a_simmx_file_path = os.path.join(dir_path, 'test_tfidf_a.simmx')

    def tearDown(self):
        # cleanup
        if os.path.isfile(self.test_dict_file_path):
            os.remove(self.test_dict_file_path)
            os.remove(self.test_md_file_path)
            os.remove(self.test_q_simmx_file_path)
            os.remove(self.test_a_simmx_file_path)

    @patch('ir_query_engine.retrieve_match_models.tf_idf_feature.tfidf_model.get_md_path')
    @patch('ir_query_engine.retrieve_match_models.tf_idf_feature.tfidf_model.get_dict_path')
    @patch('ir_query_engine.retrieve_match_models.tf_idf_feature.tfidf_model.get_a_simmx_path')
    @patch('ir_query_engine.retrieve_match_models.tf_idf_feature.tfidf_model.get_q_simmx_path')
    def test_get_model(self, mock_q_simmx_file_path, mock_a_simmx_file_path, mock_dict_file_path, mock_md_file_path):
        mock_md_file_path.return_value = self.test_md_file_path
        mock_dict_file_path.return_value = self.test_dict_file_path
        mock_q_simmx_file_path.return_value = self.test_q_simmx_file_path
        mock_a_simmx_file_path.return_value = self.test_a_simmx_file_path

        new_model_struct = TfIdfModelStruct.get_model(data_store=self.data_store)
        # assert file saves
        self.assertTrue(os.path.isfile(self.test_md_file_path))
        self.assertTrue(os.path.isfile(self.test_dict_file_path))
        self.assertTrue(os.path.isfile(self.test_q_simmx_file_path))
        self.assertTrue(os.path.isfile(self.test_a_simmx_file_path))

        loaded_model_struct = TfIdfModelStruct.get_model()
        # assert models loaded identity
        self.assertEqual(new_model_struct.dictionary,
                         loaded_model_struct.dictionary)

        bow = [(0, 1), (4, 1)]
        self.assertEqual(new_model_struct.model[bow],
                         loaded_model_struct.model[bow])
        vec = new_model_struct.model[bow]
        self.assertEqual(new_model_struct.answer_sim_matrix[vec][0],
                         loaded_model_struct.answer_sim_matrix[vec][0])
        self.assertEqual(new_model_struct.question_sim_matrix[vec][0],
                         loaded_model_struct.question_sim_matrix[vec][0])

        # regen
        self.data_store.doc_set[0] = "Yo man"
        regen_model_struct = TfIdfModelStruct.get_model(data_store=self.data_store, regen=True)
        self.assertNotEqual(new_model_struct.dictionary,
                            regen_model_struct.dictionary)

    @patch('ir_query_engine.retrieve_match_models.tf_idf_feature.tfidf_model.get_md_path')
    @patch('ir_query_engine.retrieve_match_models.tf_idf_feature.tfidf_model.get_dict_path')
    @patch('ir_query_engine.retrieve_match_models.tf_idf_feature.tfidf_model.get_a_simmx_path')
    @patch('ir_query_engine.retrieve_match_models.tf_idf_feature.tfidf_model.get_q_simmx_path')
    def test_query(self, mock_q_simmx_file_path, mock_a_simmx_file_path, mock_dict_file_path, mock_md_file_path):
        mock_md_file_path.return_value = self.test_md_file_path
        mock_dict_file_path.return_value = self.test_dict_file_path
        mock_q_simmx_file_path.return_value = self.test_q_simmx_file_path
        mock_a_simmx_file_path.return_value = self.test_a_simmx_file_path

        model_struct = TfIdfModelStruct.get_model(data_store=self.data_store)

        query_doc = "Is brocolli tasty to eat?"
        results = model_struct.query_answers(raw_doc=query_doc)
        # The ids in the results tuple are the positions in the answer corpus,
        # i.e. the index in the answer_set, translate back to doc_id
        results = self.data_store.translate_answer_query_results(results)

        # the most similar one is the first answer
        self.assertEqual(results[0][0], 1)

        results = model_struct.query_questions(raw_doc=query_doc)
        # The ids in the results tuple are the positions in the answer corpus,
        # i.e. the index in the answer_set, translate back to doc_id
        results = self.data_store.translate_question_query_results(results)

        self.assertEqual(results[0][0], 0)

    @patch('ir_query_engine.retrieve_match_models.tf_idf_feature.tfidf_model.get_md_path')
    @patch('ir_query_engine.retrieve_match_models.tf_idf_feature.tfidf_model.get_dict_path')
    @patch('ir_query_engine.retrieve_match_models.tf_idf_feature.tfidf_model.get_a_simmx_path')
    @patch('ir_query_engine.retrieve_match_models.tf_idf_feature.tfidf_model.get_q_simmx_path')
    def test_td_idf(self, mock_q_simmx_file_path, mock_a_simmx_file_path, mock_dict_file_path, mock_md_file_path):
        mock_md_file_path.return_value = self.test_md_file_path
        mock_dict_file_path.return_value = self.test_dict_file_path
        mock_q_simmx_file_path.return_value = self.test_q_simmx_file_path
        mock_a_simmx_file_path.return_value = self.test_a_simmx_file_path

        model_struct = TfIdfModelStruct.get_model(data_store=self.data_store)

        query_doc = "Is brocolli tasty to eat?"
        lowered_doc = query_doc.lower()
        tf_vec = model_struct.get_tf_vec(query_doc)
        for termid, count in tf_vec:
            self.assertEqual(count, 1)
            term = model_struct.dictionary.get(termid)
            self.assertTrue(term in lowered_doc)

        idf_vec = model_struct.get_idf_vec(tf_vec=tf_vec)
        for termid, idf in idf_vec:
            term = model_struct.dictionary.get(termid)
            df = 0
            for doc in self.data_store.doc_set:
                if term in doc.lower():
                    df += 1
            self.assertAlmostEqual(df2idf(df, len(self.data_store.doc_set)),
                                   idf)

        tf, idf = model_struct.get_tf_and_idf(query_doc, 'to')
        self.assertEqual(tf, 1)
        self.assertAlmostEqual(idf, df2idf(4, len(self.data_store.doc_set)))

        tf, idf = model_struct.get_tf_and_idf(query_doc, 'AAA')
        self.assertEqual(tf, 0)
        self.assertEqual(idf, 10.0)

        # test cooccurance features
        doc2 = self.data_store.doc_set[1]
        coocur_size, coocur_rate, cooccur_sum_idf, cooccur_avg_idf = \
            model_struct.get_cooccur_features(query_doc, doc2)

        self.assertEqual(coocur_size, 4)
        self.assertAlmostEqual(coocur_rate, float(4) / len(model_struct.get_tf_vec(doc2)))
        expected_sum = 0
        for t in model_struct.get_idf_vec(raw_doc=query_doc):
            expected_sum += t[1]

        self.assertAlmostEqual(cooccur_sum_idf, expected_sum)
        self.assertAlmostEqual(cooccur_avg_idf, expected_sum / 4)

    @patch('ir_query_engine.retrieve_match_models.tf_idf_feature.tfidf_model.get_md_path')
    @patch('ir_query_engine.retrieve_match_models.tf_idf_feature.tfidf_model.get_dict_path')
    @patch('ir_query_engine.retrieve_match_models.tf_idf_feature.tfidf_model.get_a_simmx_path')
    @patch('ir_query_engine.retrieve_match_models.tf_idf_feature.tfidf_model.get_q_simmx_path')
    def test_get_similarities(self, mock_q_simmx_file_path, mock_a_simmx_file_path, mock_dict_file_path, mock_md_file_path):
        mock_md_file_path.return_value = self.test_md_file_path
        mock_dict_file_path.return_value = self.test_dict_file_path
        mock_q_simmx_file_path.return_value = self.test_q_simmx_file_path
        mock_a_simmx_file_path.return_value = self.test_a_simmx_file_path

        model_struct = TfIdfModelStruct.get_model(data_store=self.data_store)

        query_doc = "Is brocolli tasty to eat?"
        compare_docs = self.data_store.doc_set

        sims = model_struct.get_similarities(query_doc, compare_docs)

        for idx, sim in enumerate(sims):
            expected_sim = cossim(
                model_struct.get_tfidf_vec(query_doc),
                model_struct.get_tfidf_vec(compare_docs[idx])
            )
            self.assertAlmostEqual(sim[1], expected_sim)



if __name__ == '__main__':
    unittest.main()
