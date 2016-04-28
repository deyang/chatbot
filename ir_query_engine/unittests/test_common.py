import unittest
from ir_query_engine.common import pre_process_doc

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

if __name__ == '__main__':
    unittest.main()
