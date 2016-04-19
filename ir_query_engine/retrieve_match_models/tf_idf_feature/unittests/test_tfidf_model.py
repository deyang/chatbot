import unittest
import os

from mock import patch
from ir_query_engine.common import DataStore
from ir_query_engine.retrieve_match_models.tf_idf_feature.transform import get_model
__author__ = 'Deyang'


class TfIdfModelTestCase(unittest.TestCase):
    def setUp(self):
        # create sample documents
        doc_a = "My mother spends a lot of time driving my brother around to baseball practice."
        doc_b = "Brocolli is good to eat. My brother likes to eat good brocolli, but not my mother."
        doc_c = "Some health experts suggest that driving may cause increased tension and blood pressure."
        doc_d = "I often feel pressure to perform well at school, but my mother never seems to drive my brother to do better."
        doc_e = "Health professionals say that brocolli is good for your health."

        # compile sample documents into a list
        doc_set = [doc_a, doc_b, doc_c, doc_d, doc_e]
        self.data_store = DataStore({})
        self.data_store.doc_set = doc_set

        dir_path = os.path.dirname(os.path.abspath(__file__))
        self.test_md_file_path = os.path.join(dir_path, 'test_tfidf.md')
        self.test_dict_file_path = os.path.join(dir_path, 'test_tfidf.dict')
        self.test_simmx_file_path = os.path.join(dir_path, 'test_tfidf.simmx')

    def tearDown(self):
        # cleanup
        os.remove(self.test_dict_file_path)
        os.remove(self.test_md_file_path)
        os.remove(self.test_simmx_file_path)

    @patch('ir_query_engine.retrieve_match_models.tf_idf_feature.transform.get_md_path')
    @patch('ir_query_engine.retrieve_match_models.tf_idf_feature.transform.get_dict_path')
    @patch('ir_query_engine.retrieve_match_models.tf_idf_feature.transform.get_simmx_path')
    def test_get_model(self, mock_simmx_file_path, mock_dict_file_path, mock_md_file_path):
        mock_md_file_path.return_value = self.test_md_file_path
        mock_dict_file_path.return_value = self.test_dict_file_path
        mock_simmx_file_path.return_value = self.test_simmx_file_path

        new_model_struct = get_model(data_store=self.data_store)
        # assert file saves
        self.assertTrue(os.path.isfile(self.test_md_file_path))
        self.assertTrue(os.path.isfile(self.test_dict_file_path))
        self.assertTrue(os.path.isfile(self.test_simmx_file_path))

        loaded_model_struct = get_model()
        # assert models loaded identity
        self.assertEqual(new_model_struct.dictionary,
                         loaded_model_struct.dictionary)

        bow = [(0, 1), (4, 1)]
        self.assertEqual(new_model_struct.model[bow],
                         loaded_model_struct.model[bow])
        vec = new_model_struct.model[bow]
        self.assertEqual(new_model_struct.sim_matrix[vec][0],
                         loaded_model_struct.sim_matrix[vec][0])

        # regen
        self.data_store.doc_set.pop()
        regen_model_struct = get_model(data_store=self.data_store, regen=True)
        self.assertNotEqual(new_model_struct.dictionary,
                            regen_model_struct.dictionary)

    @patch('ir_query_engine.retrieve_match_models.tf_idf_feature.transform.get_md_path')
    @patch('ir_query_engine.retrieve_match_models.tf_idf_feature.transform.get_dict_path')
    @patch('ir_query_engine.retrieve_match_models.tf_idf_feature.transform.get_simmx_path')
    def test_query(self, mock_simmx_file_path, mock_dict_file_path, mock_md_file_path):
        mock_md_file_path.return_value = self.test_md_file_path
        mock_dict_file_path.return_value = self.test_dict_file_path
        mock_simmx_file_path.return_value = self.test_simmx_file_path

        model_struct = get_model(data_store=self.data_store)

        query_doc = "Is brocolli tasty to eat?"
        results = model_struct.query(raw_doc=query_doc)
        # the most similar one is the first doc
        self.assertEqual(results[0][0], 1)


if __name__ == '__main__':
    unittest.main()
