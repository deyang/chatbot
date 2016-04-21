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

            # Similar concept from Google
            #TODO

        return training_qa_pairs

    @staticmethod
    def __get_comparison_score_between(concept_a, concept_b):
        if isinstance(concept_a, EntityClassLevelQA):
            if isinstance(concept_b, EntityClassLevelQA):
                if concept_a.entity_class == concept_b.entity_class:
                    return 100
                else:
                    # same class level query on a different entity type
                    return 30
            else:
                if concept_a.entity_class == concept_b.entity_instance.__class__:
                    # broader query on the same entity type
                    return 60
                else:
                    # broader query on a different entity type
                    return 20
        elif isinstance(concept_a, EntityInstanceSelfQA):
            if isinstance(concept_b, EntityClassLevelQA):
                if concept_a.entity_instance.__class__ == concept_b.entity_class:
                    # broader query on the same entity type
                    return 60
                else:
                    # broader query on a different entity type
                    return 20
            elif isinstance(concept_b, EntityInstanceSelfQA):
                if concept_a.entity_instance == concept_b.entity_instance:
                    return 100
                else:
                    # same property, different entity
                    return 40
            else:
                if concept_a.entity_instance == concept_b.entity_instance:
                    # same entity, different property
                    return 50
                else:
                    # different entity, different property
                    return 10
        elif isinstance(concept_a, EntityPropertyQA):
            if isinstance(concept_b, EntityClassLevelQA):
                if concept_a.entity_instance.__class__ == concept_b.entity_class:
                    # broader query on the same entity type
                    return 60
                else:
                    # broader query on a different entity type
                    return 20
            elif isinstance(concept_b, EntityPropertyQA):
                same_entity = concept_a.entity_instance == concept_b.entity_instance
                same_property = concept_a.property_def == concept_b.property_def
                if same_entity and same_property:
                    return 100
                elif same_entity and not same_property:
                    return 50
                elif not same_entity and same_property:
                    return 40
                else:
                    return 10
            else:
                if concept_a.entity_instance == concept_b.entity_instance:
                    # same entity, different property
                    return 50
                else:
                    # different entity, different property
                    return 10
        elif isinstance(concept_a, EntityRelationQA):
            if isinstance(concept_b, EntityClassLevelQA):
                if concept_a.entity_instance.__class__ == concept_b.entity_class:
                    # broader query on the same entity type
                    return 60
                else:
                    # broader query on a different entity type
                    return 20
            elif isinstance(concept_b, EntityRelationQA):
                same_entity = concept_a.entity_instance == concept_b.entity_instance
                same_relation = concept_a.relation_def == concept_b.relation_def
                if same_entity and same_relation:
                    return 100
                elif same_entity and not same_relation:
                    return 50
                elif not same_entity and same_relation:
                    return 40
                else:
                    return 10
            else:
                if concept_a.entity_instance == concept_b.entity_instance:
                    # same entity, different relation
                    return 50
                else:
                    # different entity, different relation
                    return 10

