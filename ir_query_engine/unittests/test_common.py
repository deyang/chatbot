import unittest
import json
from mock import patch, mock_open, call
from ir_query_engine.common import pre_process_doc, split_data_to_train_validate_test

__author__ = 'Deyang'


class CommonTestCase(unittest.TestCase):

    def test_pre_process(self):
        raw_doc = 'You can find the website of Matternet at http'
        self.assertEqual(pre_process_doc(raw_doc),
                         [u'you', u'can', u'find', u'the', u'websit', u'of', u'matternet', u'at', u'http'])
        raw_doc = 'What do you like to eat?'
        self.assertEqual(pre_process_doc(raw_doc),
                         [u'what', u'do', u'you', u'like', u'to', u'eat'])
        self.assertEqual(pre_process_doc(raw_doc, rm_stop_words=True),
                         [u'like', u'eat'])

    @patch('ir_query_engine.common.load_raw_data')
    def test_split_data_to_train_validate_test(self, mock_load_raw_data):
        input_object = [
            {"_question": "q%s" % i, "answer": "a%s" % i} for i in range(10)
        ]
        mock_load_raw_data.return_value = input_object
        m = mock_open()
        with patch('ir_query_engine.common.open', m, create=True):
            split_data_to_train_validate_test("input.json", 0.6, 0.2)
            calls = [call('/Users/Deyang/Work/chatbot/ir_query_engine/../data/input_train.json', 'w'),
                     call('/Users/Deyang/Work/chatbot/ir_query_engine/../data/input_validate.json', 'w'),
                     call('/Users/Deyang/Work/chatbot/ir_query_engine/../data/input_test.json', 'w')]
            m.assert_has_calls(calls, any_order=True)

            handle = m()
            train_content = handle.write.call_args_list[0][0][0]
            train_object = json.loads(train_content)
            validate_content = handle.write.call_args_list[1][0][0]
            validate_object = json.loads(validate_content)
            test_content = handle.write.call_args_list[2][0][0]
            test_object = json.loads(test_content)

            self.assertEqual(len(train_object), 6)
            self.assertEqual(len(validate_object), 2)
            self.assertEqual(len(test_object), 2)

            recover_data = train_object + validate_object + test_object
            self.assertEqual(recover_data, input_object)


if __name__ == '__main__':
    unittest.main()
