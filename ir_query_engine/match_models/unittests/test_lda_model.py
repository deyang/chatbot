import unittest

import os
from mock import patch
from ir_query_engine.common import DataStore
from ir_query_engine.match_models.lda_model import LdaModelStruct

__author__ = 'Deyang'


class LdaModelTestCase(unittest.TestCase):
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
                "answer": "Some health experts suggest that driving may cause increased tension and blood pressure."
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
        self.test_md_file_path = os.path.join(dir_path, 'test_lda.md')
        self.test_dict_file_path = os.path.join(dir_path, 'test_lda.dict')
        self.test_simmx_file_path = os.path.join(dir_path, 'test_lda.simmx')
        self.test_num_topcis_file_path = os.path.join(dir_path, 'test_lda_num_topics.txt')

    def tearDown(self):
        # cleanup
        os.remove(self.test_md_file_path)
        os.remove(self.test_dict_file_path)
        os.remove(self.test_simmx_file_path)
        os.remove(self.test_num_topcis_file_path)
        os.remove(".".join((self.test_md_file_path, "state")))

    @patch('ir_query_engine.match_models.lda_model.get_num_topic_path')
    @patch('ir_query_engine.match_models.lda_model.get_md_path')
    @patch('ir_query_engine.match_models.lda_model.get_dict_path')
    @patch('ir_query_engine.match_models.lda_model.get_simmx_path')
    def test_get_model(self, mock_simmx_file_path, mock_dict_file_path, mock_md_file_path, mock_num_topics_path):
        mock_md_file_path.return_value = self.test_md_file_path
        mock_dict_file_path.return_value = self.test_dict_file_path
        mock_simmx_file_path.return_value = self.test_simmx_file_path
        mock_num_topics_path.return_value = self.test_num_topcis_file_path

        new_model_struct = LdaModelStruct.get_model(data_store=self.data_store, num_topics=2)
        # assert file saves
        self.assertTrue(os.path.isfile(self.test_md_file_path))
        self.assertTrue(os.path.isfile(self.test_dict_file_path))
        self.assertTrue(os.path.isfile(self.test_simmx_file_path))
        self.assertTrue(os.path.isfile(self.test_num_topcis_file_path))

        loaded_model_struct = LdaModelStruct.get_model()
        self.assertEqual(loaded_model_struct.num_topics, 2)
        self.assertEqual(new_model_struct.dictionary,
                         loaded_model_struct.dictionary)

        bow = [(0, 1), (4, 1)]
        self.assertAlmostEqual(new_model_struct.model[bow][0][1],
                               loaded_model_struct.model[bow][0][1],
                               delta=0.001)
        self.assertAlmostEqual(new_model_struct.model[bow][1][1],
                               loaded_model_struct.model[bow][1][1],
                               delta=0.001)
        predict = new_model_struct.model[bow]
        self.assertAlmostEqual(new_model_struct.sim_matrix[predict][0],
                               loaded_model_struct.sim_matrix[predict][0])

        # regen
        self.data_store.doc_set = self.data_store.doc_set[1:]
        regen_model_struct = LdaModelStruct.get_model(data_store=self.data_store, regen=True, num_topics=2)
        self.assertNotEqual(new_model_struct.dictionary,
                            regen_model_struct.dictionary)

    @patch('ir_query_engine.match_models.lda_model.get_num_topic_path')
    @patch('ir_query_engine.match_models.lda_model.get_md_path')
    @patch('ir_query_engine.match_models.lda_model.get_dict_path')
    @patch('ir_query_engine.match_models.lda_model.get_simmx_path')
    def test_query(self, mock_simmx_file_path, mock_dict_file_path, mock_md_file_path, mock_num_topics_path):
        mock_md_file_path.return_value = self.test_md_file_path
        mock_dict_file_path.return_value = self.test_dict_file_path
        mock_simmx_file_path.return_value = self.test_simmx_file_path
        mock_num_topics_path.return_value = self.test_num_topcis_file_path

        model_struct = LdaModelStruct.get_model(data_store=self.data_store, num_topics=3)

        query_doc = "Eat well. Be Heathy. Brocolli"
        results = model_struct.query(raw_doc=query_doc)
        self.assertNotEqual(results[0][0], 3)

        results = model_struct.get_similarities(query_doc, [query_doc, self.data_store.doc_set[0]])
        self.assertAlmostEqual(results[0][1], 1.0, delta=0.001)


if __name__ == '__main__':
    unittest.main()
