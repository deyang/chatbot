import random
import math
from qagen.qa.qa_pair import *


class TrainingLabelGenerator(object):

    def __init__(self, unlabeled_qa_pairs, qa_concept_provider):
        self.unlabeled_qa_pairs = unlabeled_qa_pairs
        self.qa_concept_provider = qa_concept_provider

    def generate_qa_pairs_with_training_label(self, sample_count=-1):

        training_qa_pairs = self.unlabeled_qa_pairs if sample_count < 0 \
            else random.sample(self.unlabeled_qa_pairs, sample_count)

        for qa_pair in training_qa_pairs:

            qa_samples = []

            # Draw samples from these concepts
            all_class_level_concepts = self.qa_concept_provider.get_all_class_level_concepts()
            all_instance_level_concepts = self.qa_concept_provider.get_all_instance_level_concepts()

            qa_concept = qa_pair.qa_concept

            # 3 from the exact same concept
            same_concept_qa_pairs = qa_concept.qa_pair_variations
            qa_samples.extend(self.__draw_random_sample_from_list(same_concept_qa_pairs, 3))

            if isinstance(qa_concept, EntityClassLevelQA):
                # 5 from same class but instance level
                qa_samples.extend(self.__draw_random_question_from_concepts(
                    self.qa_concept_provider.get_instance_level_concepts_by_entity_type(qa_concept.entity_class), 5))
                # 7 class level, but different class
                qa_samples.extend(self.__draw_random_question_from_concepts(
                    all_class_level_concepts, 7))
                # 10 random instance level
                qa_samples.extend(self.__draw_random_question_from_concepts(
                    all_instance_level_concepts, 10))

            elif isinstance(qa_concept, EntityInstanceSelfQA):
                # 2 from same class, but class level
                qa_samples.extend(self.__draw_random_question_from_concepts(
                    self.qa_concept_provider.get_class_level_concepts_by_entity_type(qa_concept.entity_instance.__class__), 2))
                # 5 from same entity, but different property
                qa_samples.extend(self.__draw_random_question_from_concepts(
                    self.qa_concept_provider.get_instance_level_concepts_by_entity_instance(qa_concept.entity_instance), 5))
                # 5 self questions about this type, but different entity instance
                qa_samples.extend(self.__draw_random_question_from_concepts(
                    self.qa_concept_provider.get_self_concepts_by_entity_type(qa_concept.entity_instance.__class__), 5))
                # 10 random instance level
                qa_samples.extend(self.__draw_random_question_from_concepts(
                    all_instance_level_concepts, 10))

            elif isinstance(qa_concept, EntityPropertyQA):
                # 2 from same class, but class level
                qa_samples.extend(self.__draw_random_question_from_concepts(
                    self.qa_concept_provider.get_class_level_concepts_by_entity_type(qa_concept.entity_instance.__class__), 2))
                # 5 from same entity, but different property
                qa_samples.extend(self.__draw_random_question_from_concepts(
                    self.qa_concept_provider.get_instance_level_concepts_by_entity_instance(qa_concept.entity_instance), 5))
                # 5 from same property (hence same class), but different entity instance
                qa_samples.extend(self.__draw_random_question_from_concepts(
                    self.qa_concept_provider.get_instance_level_concepts_by_property_def(qa_concept.property_def), 5))
                # 10 random instance level
                qa_samples.extend(self.__draw_random_question_from_concepts(
                    all_instance_level_concepts, 10))

            elif isinstance(qa_concept, EntityRelationQA):
                # 2 from same class, but class level
                qa_samples.extend(self.__draw_random_question_from_concepts(
                    self.qa_concept_provider.get_class_level_concepts_by_entity_type(qa_concept.entity_instance.__class__), 2))
                # 5 from same entity, but different property
                qa_samples.extend(self.__draw_random_question_from_concepts(
                    self.qa_concept_provider.get_instance_level_concepts_by_entity_instance(qa_concept.entity_instance), 5))
                # 5 from same relation (hence same class), but different entity instance
                qa_samples.extend(self.__draw_random_question_from_concepts(
                    self.qa_concept_provider.get_instance_level_concepts_by_relation_def(qa_concept.relation_def), 5))
                # 10 random instance level
                qa_samples.extend(self.__draw_random_question_from_concepts(
                    all_instance_level_concepts, 10))

            else:
                raise Exception('Unexpected codepath')

            qa_samples = self.__dedup_qa_pairs(qa_samples)

            # add the scores of the sample to the qa_pair
            for qa_sample in qa_samples:
                score = self.__get_comparison_score_between(qa_sample.qa_concept, qa_pair.qa_concept)
                qa_pair.add_qa_pair_with_matching_score(qa_sample.question, qa_sample.answer, score)

            # end-of-for-loop

        return training_qa_pairs

    @staticmethod
    def __get_comparison_score_between(concept_a, concept_b):
        return concept_a.compare_score_with(concept_b)

    @staticmethod
    def __draw_random_question_from_concepts(concept_list, question_count):

        question_per_concept = 1
        if len(concept_list) < question_count:
            question_per_concept = int(math.floor(question_count / len(concept_list)))

        all_questions = []
        sample_concepts = TrainingLabelGenerator.__draw_random_sample_from_list(concept_list, question_count)
        for concept in sample_concepts:
            all_questions.extend(TrainingLabelGenerator.__draw_random_sample_from_list(
               concept.qa_pair_variations, question_per_concept
            ))
        return all_questions

    @staticmethod
    def __draw_random_sample_from_list(source_list, sample_size):
        sample_size = min(sample_size, len(source_list))
        return random.sample(source_list, sample_size)

    @staticmethod
    def __dedup_qa_pairs(qa_list):
        deduped = []
        question_set = set()
        for qa in qa_list:
            if qa.question not in question_set:
                question_set.add(qa.question)
                deduped.append(qa)
        return deduped
