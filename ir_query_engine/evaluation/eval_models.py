from ir_query_engine.query_engine import QueryEngine
from ir_query_engine.common import load_data


from optparse import OptionParser

__author__ = 'Deyang'

if __name__ == '__main__':

    parser = OptionParser()
    parser.add_option('-d', '--data_file', dest='data_file',
                      action='store',
                      default=None,
                      help='Input data file')

    (options, args) = parser.parse_args()

    data_store = load_data(options.data_file)
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
