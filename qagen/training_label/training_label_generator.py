import random
from qagen.qa.qa_pair import *

SAME_CONCEPT_QA_PAIR_FROM_TRAINING_SET = 3
OTHER_CONCEPT_QA_PAIR_FROM_TRAINING_SET = 100
SIMILAR_SURFACE_QA_PAIR_FROM_WEB = 10

MAX_SCORE = 100


class TrainingLabelGenerator(object):

    def __init__(self, unlabeled_qa_pairs):
        self.unlabeled_qa_pairs = unlabeled_qa_pairs

    def generate_qa_pairs_with_training_label(self, sample_count=-1):

        training_qa_pairs = self.unlabeled_qa_pairs if sample_count < 0 \
            else random.sample(self.unlabeled_qa_pairs, sample_count)

        for qa_pair in training_qa_pairs:

            # Add QAs under the same concept with a max score
            qa_concept = qa_pair.qa_concept
            same_concept_qa_pairs = qa_concept.qa_pair_variations
            for qa_sample in random.sample(
                same_concept_qa_pairs, min(len(same_concept_qa_pairs), SAME_CONCEPT_QA_PAIR_FROM_TRAINING_SET)
            ):
                qa_pair.add_qa_pair_with_matching_score(qa_sample.question, qa_sample.answer, MAX_SCORE)

            # Add known QA pairs under different concepts
            for qa_sample in random.sample(self.unlabeled_qa_pairs, OTHER_CONCEPT_QA_PAIR_FROM_TRAINING_SET):
                score = self.__get_comparison_score_between(qa_sample.qa_concept, qa_pair.qa_concept)
                # print '%s  +  %s = %d' % (qa_sample.qa_concept, qa_pair.qa_concept, score)
                qa_pair.add_qa_pair_with_matching_score(qa_sample.question, qa_sample.answer, score)

        return training_qa_pairs

    @staticmethod
    def __get_comparison_score_between(concept_a, concept_b):
        return concept_a.compare_score_with(concept_b)

