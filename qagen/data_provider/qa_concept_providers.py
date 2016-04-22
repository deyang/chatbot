from qagen.qa.qa_pair import EntityClassLevelQA, EntityInstanceSelfQA, EntityPropertyQA, EntityRelationQA

K_ALL_CLASS_LEVEL_CONCEPTS = 'all_class_level_concepts'
K_ENTITY_TYPE_TO_CLASS_LEVEL_CONCEPT = 'entity_type_to_class_level_concept'

K_ALL_INSTANCE_LEVEL_CONCEPTS = 'all_instance_level_concepts'
K_ENTITY_TYPE_TO_INSTANCE_LEVEL_CONCEPT = 'entity_type_to_instance_level_concept'
K_ENTITY_INSTANCE_TO_INSTANCE_LEVEL_CONCEPT = 'entity_instance_to_instance_level_concept'
K_PROPERTY_AND_RELATION_DEF_TO_CONCEPT = 'property_and_relation_def_to_concept'
K_SELF_PROPERTY = 'self_property'

class DefaultQAConceptProvider(object):
    """
    QA concept provider backed by a knowledge data provider.
    """

    def __init__(self, data_provider):
        """
        :param data_provider: an instance of KnowledgeDataProvider
        """
        print 'Expanding QA concepts...'
        self.qa_concepts = self.__expand_qa_concepts(data_provider)

        print 'Categorizing QA concepts...'
        self.qa_concepts_by_type = self.__organize_concepts_by_type(self.qa_concepts)

    @staticmethod
    def __expand_qa_concepts(data_provider):

        concepts = []

        for entity_class in data_provider.get_all_entity_types():
            # class level
            concepts.append(EntityClassLevelQA(entity_class))

            for entity_instance in data_provider.get_all_instances_of_type(entity_class):
                # instance self
                concepts.append(EntityInstanceSelfQA(entity_instance))
                # instance property
                for property_name, property_def in entity_class.property_def_map.iteritems():
                    concepts.append(EntityPropertyQA(entity_instance, property_def))
                # instance relation
                for relation_name, relation_def in entity_class.relation_def_map.iteritems():
                    concepts.append(EntityRelationQA(entity_instance, relation_def))

        return concepts

    def __organize_concepts_by_type(self, concepts):

        all_c = []
        all_i = []
        c_to_c = {}
        c_to_i = {}
        i_to_i = {}
        pr_to_i = {}

        result = {
            K_ALL_CLASS_LEVEL_CONCEPTS: all_c,
            K_ALL_INSTANCE_LEVEL_CONCEPTS: all_i,

            K_ENTITY_TYPE_TO_CLASS_LEVEL_CONCEPT: c_to_c,
            K_ENTITY_TYPE_TO_INSTANCE_LEVEL_CONCEPT: c_to_i,
            K_ENTITY_INSTANCE_TO_INSTANCE_LEVEL_CONCEPT: i_to_i,
            K_PROPERTY_AND_RELATION_DEF_TO_CONCEPT: pr_to_i,
        }

        for concept in concepts:
            if isinstance(concept, EntityClassLevelQA):
                all_c.append(concept)
                self.__append_to_dict_value(c_to_c, concept.entity_class, concept)
            elif isinstance(concept, EntityInstanceSelfQA):
                all_i.append(concept)
                self.__append_to_dict_value(c_to_i, concept.entity_instance.__class__, concept)
                self.__append_to_dict_value(i_to_i, concept.entity_instance, concept)
                self.__append_to_dict_value(pr_to_i, K_SELF_PROPERTY, concept)
            elif isinstance(concept, EntityPropertyQA) and not concept.property_def.is_hidden:
                # ignore hidden property
                all_i.append(concept)
                self.__append_to_dict_value(c_to_i, concept.entity_instance.__class__, concept)
                self.__append_to_dict_value(i_to_i, concept.entity_instance, concept)
                self.__append_to_dict_value(pr_to_i, concept.property_def, concept)
            elif isinstance(concept, EntityRelationQA):
                all_i.append(concept)
                self.__append_to_dict_value(c_to_i, concept.entity_instance.__class__, concept)
                self.__append_to_dict_value(i_to_i, concept.entity_instance, concept)
                self.__append_to_dict_value(pr_to_i, concept.relation_def, concept)

        return result

    def __append_to_dict_value(self, dict, key, list_member):
        value = dict.get(key)
        if value is not None:
            value.append(list_member)
        else:
            value = [list_member]
            dict[key] = value

    def get_all_qa_concepts(self):
        return list(self.qa_concepts)

    def get_all_class_level_concepts(self):
        return self.qa_concepts_by_type[K_ALL_CLASS_LEVEL_CONCEPTS]

    def get_all_instance_level_concepts(self):
        return self.qa_concepts_by_type[K_ALL_INSTANCE_LEVEL_CONCEPTS]

    def get_class_level_concepts_by_entity_type(self, entity_class):
        return self.qa_concepts_by_type[K_ENTITY_TYPE_TO_CLASS_LEVEL_CONCEPT][entity_class]

    def get_instance_level_concepts_by_entity_type(self, entity_class):
        return self.qa_concepts_by_type[K_ENTITY_TYPE_TO_INSTANCE_LEVEL_CONCEPT][entity_class]

    def get_instance_level_concepts_by_entity_instance(self, entity_instance):
        return self.qa_concepts_by_type[K_ENTITY_INSTANCE_TO_INSTANCE_LEVEL_CONCEPT][entity_instance]

    def get_instance_level_concepts_by_property_def(self, property_def):
        return self.qa_concepts_by_type[K_PROPERTY_AND_RELATION_DEF_TO_CONCEPT][property_def]

    def get_instance_level_concepts_by_relation_def(self, relation_def):
        return self.qa_concepts_by_type[K_PROPERTY_AND_RELATION_DEF_TO_CONCEPT][relation_def]

    def get_all_self_concepts(self):
        return self.qa_concepts_by_type[K_PROPERTY_AND_RELATION_DEF_TO_CONCEPT][K_SELF_PROPERTY]
