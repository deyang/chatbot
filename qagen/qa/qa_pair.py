from qagen.qa.utils import tokenize_sentence, intersect_lists


class QAConcept(object):
    """
    A QAConcept represents a group of QAPair variations asking about the same fact-based concept.
    """

    def __init__(self):
        self.qa_pair_variations = []

    def new_qa_pair(self, question, answer, context_map):
        """
        returns a new QA pair associated with the current QA concept
        """
        qa_pair = QAPair(self, question, answer, context_map)
        self.qa_pair_variations.append(qa_pair)
        return qa_pair


class EntityClassLevelQA(QAConcept):

    def __init__(self, entity_class):
        super(EntityClassLevelQA, self).__init__()
        self.entity_class = entity_class

    def __repr__(self):
        return '%s.class' % self.entity_class.__name__

    def get_topic_words(self):
        return [self.entity_class.__name__.lower()]

    def compare_score_with(self, other):
        if isinstance(other, EntityClassLevelQA):
            if self.entity_class == other.entity_class:
                return 100
            else:
                # same class level query on a different entity type
                return 30
        else:
            if self.entity_class == other.entity_instance.__class__:
                # broader query on the same entity type
                return 60
            else:
                # broader query on a different entity type
                return 20


class EntityInstanceSelfQA(QAConcept):

    def __init__(self, entity_instance):
        super(EntityInstanceSelfQA, self).__init__()
        self.entity_instance = entity_instance

    def __repr__(self):
        return '%s->self' % self.entity_instance

    def get_topic_words(self):
        result = []
        result.extend(tokenize_sentence(self.entity_instance.get_entity_id().lower()))
        return result

    def compare_score_with(self, other):
        if isinstance(other, EntityClassLevelQA):
            if self.entity_instance.__class__ == other.entity_class:
                # broader query on the same entity type
                return 60
            else:
                # broader query on a different entity type
                return 20
        elif isinstance(other, EntityInstanceSelfQA):
            if self.entity_instance == other.entity_instance:
                return 100
            else:
                # same property, different entity
                return 40
        else:
            if self.entity_instance == other.entity_instance:
                # same entity, different property
                return 50
            else:
                # different entity, different property
                return 10


class EntityPropertyQA(QAConcept):

    def __init__(self, entity_instance, property_def):
        super(EntityPropertyQA, self).__init__()
        self.entity_instance = entity_instance
        self.property_def = property_def

    def __repr__(self):
        return '%s->%s' % (self.entity_instance, self.property_def.property_name)

    def get_topic_words(self):
        result = []
        result.extend(tokenize_sentence(self.entity_instance.get_entity_id().lower()))
        result.extend(tokenize_sentence(self.property_def.property_name.lower()))
        return result

    def compare_score_with(self, other):
        if isinstance(other, EntityClassLevelQA):
            if self.entity_instance.__class__ == other.entity_class:
                # broader query on the same entity type
                return 60
            else:
                # broader query on a different entity type
                return 20
        elif isinstance(other, EntityPropertyQA):
            same_entity = self.entity_instance == other.entity_instance
            same_property = self.property_def == other.property_def
            if same_entity and same_property:
                return 100
            elif same_entity and not same_property:
                return 50
            elif not same_entity and same_property:
                return 40
            else:
                return 10
        else:
            if self.entity_instance == other.entity_instance:
                # same entity, different property
                return 50
            else:
                # different entity, different property
                return 10


class EntityRelationQA(QAConcept):

    def __init__(self, entity_instance, relation_def):
        super(EntityRelationQA, self).__init__()
        self.entity_instance = entity_instance
        self.relation_def = relation_def

    def __repr__(self):
        return '%s->%s' % (self.entity_instance, self.relation_def.relation_name)

    def get_topic_words(self):
        result = []
        result.extend(tokenize_sentence(self.entity_instance.get_entity_id().lower()))
        result.extend(tokenize_sentence(self.relation_def.relation_name.lower()))
        return result

    def compare_score_with(self, other):
        if isinstance(other, EntityClassLevelQA):
            if self.entity_instance.__class__ == other.entity_class:
                # broader query on the same entity type
                return 60
            else:
                # broader query on a different entity type
                return 20
        elif isinstance(other, EntityRelationQA):
            same_entity = self.entity_instance == other.entity_instance
            same_relation = self.relation_def == other.relation_def
            if same_entity and same_relation:
                return 100
            elif same_entity and not same_relation:
                return 50
            elif not same_entity and same_relation:
                return 40
            else:
                return 10
        else:
            if self.entity_instance == other.entity_instance:
                # same entity, different relation
                return 50
            else:
                # different entity, different relation
                return 10


class QAPair(object):
    """
    A specific variation of QA pair
    """

    def __init__(self, qa_concept, question, answer, context_list):
        """
        :param qa_concept: the parent QA concept associated with this QA pair.
        :param question:
        :param answer:
        :param context_list: [{entity_class_name: entity_instance_id}]
        :return:
        """
        self.qa_concept = qa_concept
        self.question = question
        self.answer = answer
        self.context_list = context_list
        self.qa_pairs_with_matching_score = []

    def __repr__(self):
        return 'Q: %s, A: %s' % (self.question, self.answer)

    def add_qa_pair_with_matching_score(self, question, answer, score):
        """
        used for generating training label
        :param score: 0~100
        """
        self.qa_pairs_with_matching_score.append({
            '_question': question,
            'answer': answer,
            'score': score
        })

    def to_json_dict(self, is_for_training=False):
        result = {
            # use _question instead of question as the keyword so that they are in alphabetic order
            '_question': self.question,
            'answer': self.answer,
            'context': self.context_list,
        }
        if is_for_training:
            result.update({
                '_question_topic_words': intersect_lists(
                    tokenize_sentence(self.question.lower()), self.qa_concept.get_topic_words()
                ),
                'answer_topic_words': intersect_lists(
                    tokenize_sentence(self.answer.lower()), self.qa_concept.get_topic_words()
                ),
                'qa_pairs_with_matching_score': self.qa_pairs_with_matching_score
            })
        return result


