import os
from sklearn.externals import joblib
from gensim import similarities
from ir_query_engine.retrieve_match_models.tf_idf_feature.tfidf_model import TfIdfModelStruct
from ir_query_engine.common import p_stemmer, pre_process_doc
import re

from ir_query_engine import engine_logger
from sklearn.linear_model import LogisticRegression
import nltk
from pylru import lrucache
from ir_query_engine.common import DataStore

__author__ = 'Deyang'

DIR_PATH = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE_PATH = os.path.join(DIR_PATH, 'lr.pkl')


def get_md_path():
    return MODEL_FILE_PATH

nltk.download('maxent_ne_chunker')
nltk.download('tagsets')


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

    @staticmethod
    def normalize_word(raw_word):
        return p_stemmer.stem(raw_word.lower())

    def analyze(self, raw_doc, stemmed_word):
        """
        The POS tagging and NE extraction is done on the raw document.
        While the result is stored as normed. So that the lookup will no miss.
        :param raw_doc:
        :param stemmed_word:
        :return:
        """
        stemmed_word = self.normalize_word(stemmed_word)

        if raw_doc in self.cache:
            # engine_logger.debug("Hit doc word analyzer cache.")
            analyze_results = self.cache[raw_doc]
        else:
            cleaned_doc = re.sub(r'https?:\/\/.*\s?$', 'http', raw_doc.lower())
            sentences = nltk.sent_tokenize(cleaned_doc)
            tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]

            sent_word_bags = []
            for tokenized_sent in tokenized_sentences:
                bag = set([])
                for token in tokenized_sent:
                    bag.add(self.normalize_word(token))
                sent_word_bags.append(bag)

            tagged_sentences = [nltk.pos_tag(sentence) for sentence in tokenized_sentences]
            word_to_pos_tag = dict()

            for tagged_sent in tagged_sentences:
                for tagged_word in tagged_sent:
                    word = self.normalize_word(tagged_word[0])
                    tag = POS_TAG_LABELS.get(tagged_word[1], None)
                    if tag is None:
                        continue
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
                        ne_word_set.add(self.normalize_word(child[0]))

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
                # It could be the first and the last the same time
                if idx == len(analyze_results.sent_word_bags) - 1:
                    LAST = 1
                    if NE:
                        NE_LAST = 1
        pos_tags = analyze_results.word_to_pos_tag.get(stemmed_word, dict())

        if len(pos_tags) == 0:
            POS = 0
        else:
            pos_tags = sorted(pos_tags.items(), key=lambda t: t[1], reverse=True)
            POS = pos_tags[0][0]

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


class TopicWordModelStruct(object):

    def __init__(self, tfidf_model_struct, model, training_doc_word_pairs=None, training_labels=None):
        # Topic model mush have a tfidf model struct
        self.tfidf_model_struct = tfidf_model_struct
        self.model = model
        self.analyzer = DocWordAnalyzer()
        self.training_doc_word_pairs = training_doc_word_pairs
        self.training_labels = training_labels
        self.training_vecs = []

    def _train(self):
        for pair in self.training_doc_word_pairs:
            self.training_vecs.append(self._extract_feature(pair))
        engine_logger.info("Training logistic regression model in topic word model")
        self.model.fit(self.training_vecs, self.training_labels)

    def get_similarities(self, query_doc, compare_docs):
        query_prob_vec = self._predict_doc_as_vec(query_doc)
        compare_prob_vecs = [self._predict_doc_as_vec(doc) for doc in compare_docs]
        sim_mx = similarities.SparseMatrixSimilarity(compare_prob_vecs, num_features=len(self.tfidf_model_struct.dictionary))
        sims = sim_mx[query_prob_vec]
        return list(enumerate(sims))

    def predict_one_word(self, feature_vec):
        # mainly for testing
        return self.model.predict_proba([feature_vec])[0][1]

    def _predict_doc_as_vec(self, raw_doc):
        """
        Get the prob sparse vec repr of a raw doc.
        The vec is represented by a list of tuple.
        Inside each tuple, the first is the termid,
        while the second is the prod of the word being a topic word

        :param raw_doc:
        :return:
        """
        tokens = pre_process_doc(raw_doc)
        feature_vecs = []
        doc_word_pairs = [(raw_doc, token) for token in tokens]
        for doc_word_pair in doc_word_pairs:
            feature_vecs.append(self._extract_feature(doc_word_pair))

        predicts = self.model.predict_proba(feature_vecs)
        # pos_probs has the exact same length of tokens
        # but the final results will miss the token not in the vocabulary
        pos_probs = [probs[1] for probs in predicts]

        # merge the prob with the token id
        prob_sparse_vec = []
        for idx, token in enumerate(tokens):
            termid = self.tfidf_model_struct.dictionary.token2id.get(token, None)
            if termid is None:
                continue
            prob_sparse_vec.append((termid, pos_probs[idx]))

        return prob_sparse_vec

    @classmethod
    def get_model(cls, tfidf_model_struct=None, data_store=None, regen=False):
        md_file_path = get_md_path()
        if not os.path.isfile(md_file_path) or regen:
            engine_logger.info("Generating topic word logistic regression model")

            # Each doc is a raw doc and normed word (lowered and stemmed)
            training_doc_word_pairs = []
            training_labels = []

            for pair in data_store.topic_word_docs:
                doc = pair[0]

                # iterate the combination of doc-word pairs and as positive and negative ex.
                tokens = pre_process_doc(doc)
                positive_topics_words = set(DocWordAnalyzer.normalize_word(w) for w in pair[1])
                for t in tokens:
                    training_doc_word_pairs.append((doc, t))
                    if t in positive_topics_words:
                        training_labels.append(1)
                    else:
                        training_labels.append(0)

            if tfidf_model_struct is None:
                tfidf_model_struct = TfIdfModelStruct.get_model(data_store=data_store, regen=True, save=False)

            # un-trained model
            model = LogisticRegression()
            instance = TopicWordModelStruct(tfidf_model_struct,
                                            model,
                                            training_doc_word_pairs=training_doc_word_pairs,
                                            training_labels=training_labels)
            instance._train()

            # saving
            joblib.dump(instance.model, md_file_path)

            return instance
        else:
            # This tfidf model is loaded outside - i.e. on the whole doc space
            engine_logger.info("Loading existing topic word models")
            model = joblib.load(md_file_path)

            return TopicWordModelStruct(tfidf_model_struct, model)

    def _extract_feature(self, doc_word_pair):
        """
        features:

        0 TF: Term frequency of w in the short text
        1 IDF: Inverse document frequency of w in the whole collection
        2 SF: Number of sentences in the short text that contain w
        3 First: Whether w exists in the first sentence
        4 Last: Whether w exists in the last sentence
        5 NE: Whether w is a named entity (NE)
        6 NE: First Whether w is NE in the first sentence
        7 NE: Last Whether w is NE in the last sentence
        8 POS: Part of speech of w

        :param doc_word_pair: The doc in doc_word_pair is a raw doc.
                              For NLP analysis
        :return:
        """
        doc = doc_word_pair[0]
        word = doc_word_pair[1]
        tf, idf = self.tfidf_model_struct.get_tf_and_idf(doc, word)

        SF, FIRST, LAST, NE, NE_FIRST, NE_LAST, POS = self.analyzer.analyze(doc, word)
        return tf, idf, SF, FIRST, LAST, NE, NE_FIRST, NE_LAST, POS

