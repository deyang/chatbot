from qagen.qa.qa_pair import EntityClassLevelQA, EntityInstanceSelfQA, EntityPropertyQA, EntityRelationQA


class DefaultQAConceptProvider(object):
    """
    QA concept provider backed by a knowledge data provider.
    """

    def __init__(self, data_provider):
        """
        :param data_provider: an instance of KnowledgeDataProvider
        """
        self.qa_concepts = self.__expand_qa_concepts(data_provider)

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

    def get_all_qa_concepts(self):
        return list(self.qa_concepts)