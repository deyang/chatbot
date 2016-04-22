from ir_query_engine.query_engine import QueryEngine
from ir_query_engine.common import load_data

import nltk
from optparse import OptionParser

__author__ = 'Deyang'

# nltk.download('maxent_ne_chunker')
# nltk.download('tagsets')


class AnalyzerRewriter(object):

    DETECT = [
        "hers", "herself," "him", "himself", "hisself",
        "it", "its" "itself", "she", "thee", "theirs"
        "them", "themselves", "they", "thou", "thy", "this",
        "that", "those", "these", "her", "his", "their", "thy"
    ]

    def __init__(self):
       pass

    def rewrite(self, sent, context):
        sent = sent.lower()
        # sent = sent.replace('a16z', 'Andreessen Horowitz')
        if context is None:
            return sent
        tokenized_sent = nltk.word_tokenize(sent)
        detect = set(self.DETECT)
        for idx, token in enumerate(tokenized_sent):
            if token in detect:
                tokenized_sent[idx] = context

        return " ".join(tokenized_sent)

    @staticmethod
    def append_context(raw_query, context):
        appended = "%s %s" % (raw_query, context)
        return appended.lower()


class ConversationEngine(object):

    FALLBACK_ANSWER = \
        "I'm sorry but I'm not sure I know the answer to your question. If it's about A16Z, could you be more specific?"

    ISSUE_QUERY_WITH_CONTEXT_INJECT_THRESHOLD = 11.0

    FALLBACK_THRESHOLD = 3.2

    def __init__(self, query_engine):
        self.query_engine = query_engine
        self.last_context = None
        self.rewriter = AnalyzerRewriter()

    def talk(self, input_msg):
        rewritten_input = self.rewriter.rewrite(
            input_msg,
            self.last_context
        )
        response = self.query_engine.execute_query(rewritten_input)

        if self.FALLBACK_THRESHOLD < response.score <= self.ISSUE_QUERY_WITH_CONTEXT_INJECT_THRESHOLD and \
                self.last_context is not None:
            print "Triggering the second query with the last context: %s %s %s" % \
                  (response.score, response.answer, self.last_context)
            rewritten_input = self.rewriter.append_context(rewritten_input, self.last_context)
            second_response = self.query_engine.execute_query(rewritten_input)
            if second_response.score > response.score:
                response = second_response

        if response.score <= self.FALLBACK_THRESHOLD:
            # left the last context untouched
            print "Triggering fallback"
            return self.FALLBACK_ANSWER

        if response.context is not None and len(response.context) > 0:
            self.last_context = response.context[-1].values()[0]
        else:
            self.last_context = None

        return response.answer

if __name__ == '__main__':

    parser = OptionParser()
    parser.add_option('-d', '--data_file', dest='data_file',
                      action='store',
                      default=None,
                      help='Input data file')

    (options, args) = parser.parse_args()

    data_store = load_data(options.data_file)
    query_engine = QueryEngine(data_store)
    conversation_engine = ConversationEngine(query_engine)

    while True:
        in_msg = raw_input()
        answer = conversation_engine.talk(in_msg)
        print "Answer> %s" % answer
