import nltk
from pylru import lrucache
from ir_query_engine.retrieve_match_models.tf_idf_feature.transform import p_stemmer

__author__ = 'Deyang'


nltk.download('maxent_ne_chunker')
nltk.download('tagsets')

# part of speech
doc_ex = """
The first Republican to serve as governor of the state, he became known as the father of the Republican Party in Kentucky.
After a well-received speech seconding the presidential nomination of Ulysses S. Grant at the 1880 Republican National Convention, he was nominated for governor in 1887.
He lost the general election that year, but won in 1895, capitalizing on divisions in the Democratic Party over the issue of free silver.
His term was marked by political struggles and violence?
He advanced the status of black citizens, but was unable to enact much of his reform agenda over a hostile Democratic majority.
He was elected by the state legislature to the U.S. Senate in 1907, when voting was deadlocked and the Democratic candidate, outgoing Governor J. C. W. Beckham, refused to withdraw in favor of a compromise candidate.
Bradley's opposition to Prohibition made him palatable to some Democratic legislators, and after two months of balloting, four of them crossed party lines to elect him. His career in the Senate was largely undistinguished.
"""



tagdict = nltk.data.load("help/tagsets/upenn_tagset.pickle")
POS_TAG_LABELS = dict()
# index from 1, if the word is not matched, use 0
for i, tag in enumerate(sorted(tagdict.keys())):
    POS_TAG_LABELS[tag] = i + 1


class DocWordAnalyzer(object):

    class DocAnalyzeResults(object):

        def __init__(self, sent_word_bags, word_to_pos_tag, ne_word_set):
            self.sent_word_bags = sent_word_bags
            self.word_to_pos_tag = word_to_pos_tag
            self.ne_word_set = ne_word_set

    def __init__(self):
        # cache sent analyze results since each doc will be analyzed against its words
        self.cache = lrucache(10)

    def analyze(self, raw_doc, stemmed_word):
        """

        :param raw_doc:
        :param stemmed_word:
        :return:
        """
        if raw_doc in self.cache:
            analyze_results = self.cache[raw_doc]
        else:
            sentences = nltk.sent_tokenize(raw_doc)
            tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]

            sent_word_bags = []
            for tokenized_sent in tokenized_sentences:
                bag = set([])
                for token in tokenized_sent:
                    bag.add(p_stemmer.stem(token.lower()))
                sent_word_bags.append(bag)

            tagged_sentences = [nltk.pos_tag(sentence) for sentence in tokenized_sentences]
            word_to_pos_tag = dict()
            for tagged_sent in tagged_sentences:
                for tagged_word in tagged_sent:
                    word = p_stemmer.stem(tagged_word[0].lower())
                    tag = POS_TAG_LABELS[tagged_word[1]]
                    if word not in word_to_pos_tag:
                        word_to_pos_tag[word] = dict()
                    if tag not in word_to_pos_tag[word]:
                        word_to_pos_tag[word][tag] = 0
                    word_to_pos_tag[word][tag] += 1


            chunked_sentences = nltk.ne_chunk_sents(tagged_sentences, binary=True)

            ne_word_set = set([])
            for tree in chunked_sentences:
                # extract NE per sentence
                resutls = self.extract_entity_names(tree)

                for entity_name in resutls:
                    for child in entity_name:
                        ne_word_set.add(child[0].lower())

            analyze_results = self.DocAnalyzeResults(sent_word_bags, word_to_pos_tag, ne_word_set)
            self.cache[raw_doc] = analyze_results

        # apply word to analyze results
        SF = 0
        FIRST = 0
        LAST = 0
        NE = 1 if stemmed_word in analyze_results.ne_word_set else 0
        NE_FIRST = 0
        NE_LAST = 0
        for idx, bag in enumerate(analyze_results.sent_word_bags):
            if stemmed_word in bag:
                SF += 1
                if idx == 0:
                    FIRST = 1
                    if NE:
                        NE_FIRST = 1
                elif idx == len(analyze_results.sent_word_bags) - 1:
                    LAST = 1
                    if NE:
                        NE_LAST = 1
        pos_tags = analyze_results.word_to_pos_tag.get(stemmed_word, dict())

        if len(pos_tags) == 0:
            POS = 0
        else:
            pos_tags = sorted(pos_tags.items(), key=lambda t: t[1], reverse=True)
            POS = pos_tags[0][1]

        return SF, FIRST, LAST, NE, NE_FIRST, NE_LAST, POS



    def extract_entity_names(self, tree):
        entity_names = []

        if hasattr(tree, 'label') and tree.label:
            if tree.label() == 'NE':
                entity_names.append(tree)
            else:
                for child in tree:
                    entity_names.extend(self.extract_entity_names(child))

        return entity_names



a = DocWordAnalyzer()
print a.analyze(doc_ex, "republican")