import itertools

from qagen.knowledge.entities import *
from qagen.qa.utils import make_context_map
from qagen.qa.qa_pair import QAPair, EntityClassLevelQA, EntityInstanceSelfQA, EntityPropertyQA, EntityRelationQA


class DefaultQAPairGenerator(object):

    def __init__(self, data_provider):
        """
        :param data_provider: the knowledge entity data provider that will be used to construct answers dependent
        all the actual data
        """
        self.data_provider = data_provider

    def generate_qa_pairs_about_concept(self, qa_concept):
        # pass the concept object around so that generated QA pairs can be associated back to the concept object
        if isinstance(qa_concept, EntityClassLevelQA):
            return self.__generate_qa_pairs_about_entity_class(qa_concept)
        elif isinstance(qa_concept, EntityInstanceSelfQA):
            return self.__generate_qa_pairs_about_entity_instance(qa_concept)
        elif isinstance(qa_concept, EntityPropertyQA):
            return self.__generate_qa_pairs_about_entity_property(qa_concept)
        elif isinstance(qa_concept, EntityRelationQA):
            return self.__generate_qa_pairs_about_entity_relation(qa_concept)
        else:
            raise Exception('Unknown QA concept ' + qa_concept)

    def __generate_qa_pairs_about_entity_class(self, entity_class_concept):
        # TODO
        return []

    def __generate_qa_pairs_about_entity_instance(self, entity_self_concept):
        entity_instance = entity_self_concept.entity_instance
        wh_type = ConceptType.get_wh_type(entity_instance.__class__.entity_concept_type)
        entity_name = entity_instance.property_value_map['name']

        qa_pairs = [
            entity_self_concept.new_qa_pair(question_text, entity_instance.get_entity_self_description(),
                                            make_context_map(entity_instance))
            for question_text in [
                '%s is %s' % (wh_type, entity_name),
                'information about %s' % entity_name,
                'tell me about %s' % entity_name,
                'show me %s' % entity_name,
                'I want to know about %s' % entity_name
            ]
        ]

        ##########################
        #      Company.self      #
        ##########################
        if isinstance(entity_instance, Company):
            qa_pairs.extend([
                entity_self_concept.new_qa_pair('what does %s do' % entity_name,
                                                entity_instance.get_entity_self_description(),
                                                make_context_map(entity_instance)),
            ])

        return qa_pairs

    def __generate_qa_pairs_about_entity_property(self, entity_property_concept):
        entity_instance = entity_property_concept.entity_instance
        property_def = entity_property_concept.property_def

        # Don't ask questions about a hidden property
        if property_def.is_hidden:
            return []

        property_wh_type = ConceptType.get_wh_type(property_def.concept_type)
        property_name = property_def.property_name
        entity_name = entity_instance.property_value_map['name']

        property_value = entity_instance.property_value_map.get(property_name)
        if property_value:
            if property_def.concept_type == ConceptType.URL:
                answer = 'You can find the %s of %s at %s' %(property_name, entity_name, property_value)
            elif property_def.property_name == 'name':
                answer = property_value
            else:
                answer = 'The %s of %s is %s.' %(property_name, entity_name, property_value)
        else:
            answer = 'Sorry the %s of %s is not found on the a16z.com website.' % (property_name, entity_name)

        qa_pairs = [
            entity_property_concept.new_qa_pair(question_text, answer, make_context_map(entity_instance))
            for question_text in [
                '%s is the %s of %s' % (property_wh_type, property_name, entity_name),
                'show me the %s of %s' % (property_name, entity_name)
            ]
        ]

        ##########################
        #         *.name         #
        ##########################
        if property_def.property_name == 'name':
            qa_pairs.extend([
                entity_property_concept.new_qa_pair('what is %s called' % entity_name,
                                                    entity_instance.property_value_map['name'],
                                                    make_context_map(entity_instance)),
            ])

        if isinstance(entity_instance, Company):
            ##########################
            #    Company.founder     #
            ##########################
            if property_def.property_name == 'founder':
                founder = entity_instance.property_value_map.get('founder')
                if founder:
                    answer = '%s is founded by %s.' % (entity_name, founder)
                else:
                    answer = 'Sorry the founder information of %s is not listed on the a16z website. ' \
                             'Please checkout their website at %s' \
                             % (entity_name, entity_instance.property_value_map['website'])
                for qa_pair in qa_pairs:
                    qa_pair.answer = answer
                qa_pairs.extend([
                    entity_property_concept.new_qa_pair(question_text, answer, make_context_map(entity_instance))
                    for question_text in [
                        'who founded %s' % entity_name,
                        'who created %s' % entity_name,
                        'who started %s' % entity_name,
                        'what is the founding team of %s' % entity_name,
                        'who are the founders of %s' % entity_name,
                        'list all the founders of %s' % entity_name,
                        'show me the founders of %s' % entity_name,
                    ]
                ])

            ##########################
            #    Company.location    #
            ##########################
            elif property_def.property_name == 'location':
                location = entity_instance.property_value_map.get('location')
                if location:
                    answer = '%s is located in %s.' % (entity_name, location)
                    for qa_pair in qa_pairs:
                        qa_pair.answer = answer
                else:
                    # keep existing answer
                    answer = qa_pairs[0].answer
                qa_pairs.extend([
                    entity_property_concept.new_qa_pair(question_text, answer, make_context_map(entity_instance))
                    for question_text in [
                        'where is %s' % entity_name,
                        'where is %s located' % entity_name,
                    ]
                ])

            ##########################
            #    Company.website     #
            ##########################
            elif property_def.property_name == 'website':
                answer = qa_pairs[0].answer
                qa_pairs.extend([
                    entity_property_concept.new_qa_pair(question_text, answer, make_context_map(entity_instance))
                    for question_text in [
                        'take me to the website of %s' % entity_name,
                        'more information about %s' % entity_name,
                        'do you have a link to the website of %s' % entity_name,
                        'show me the link to the website of %s' % entity_name,
                        'I want to checkout more about %s' % entity_name,
                    ]
                ])

            ##################################
            #    Company.type of business    #
            ##################################
            elif property_def.property_name == 'type of business':
                type_of_business = entity_instance.property_value_map.get('type of business')
                if type_of_business:
                    answer = type_of_business.lower().capitalize()
                    for qa_pair in qa_pairs:
                        qa_pair.answer = answer
                else:
                    # keep existing answer
                    answer = qa_pairs[0].answer
                qa_pairs.extend([
                    entity_property_concept.new_qa_pair(question_text, answer, make_context_map(entity_instance))
                    for question_text in [
                        'what kind of business does %s do' % entity_name,
                        'what is the industry of %s' % entity_name,
                        'in what industry does %s work on' % entity_name,
                        'in what area does %s work on' % entity_name,
                        'what kind of problem does %s solve' % entity_name,
                    ]
                ])

            ##################################
            #     Company.business model     #
            ##################################
            elif property_def.property_name == 'business model':
                business_model = entity_instance.property_value_map.get('business model')
                if business_model == 'to consumer':
                    answer = '%s is a b2c (business-to-consumer) company.' % entity_name
                elif business_model == 'to enterprise':
                    answer = '%s is a b2b (business-to-business) company.' % entity_name
                else:
                    # keep existing answer
                    answer = qa_pairs[0].answer
                for qa_pair in qa_pairs:
                    qa_pair.answer = answer
                qa_pairs.extend([
                    entity_property_concept.new_qa_pair(question_text, answer, make_context_map(entity_instance))
                    for question_text in [
                        'is %s a 2b or 2c company' % entity_name,
                        'is %s a to-business or to-consumer company' % entity_name,
                        'what is target market of %s' % entity_name,
                        'who are the customers of %s' % entity_name,
                    ]
                ])
                # additional yes/no questions
                if business_model == 'to consumer':
                    qa_pairs.extend([
                        entity_property_concept.new_qa_pair(
                            'is %s a 2b company' % entity_name, 'no', make_context_map(entity_instance)),
                        entity_property_concept.new_qa_pair(
                            'is %s a 2c company' % entity_name, 'yes', make_context_map(entity_instance)),
                    ])
                elif business_model == 'to enterprise':
                    qa_pairs.extend([
                        entity_property_concept.new_qa_pair(
                            'is %s a 2b company' % entity_name, 'yes', make_context_map(entity_instance)),
                        entity_property_concept.new_qa_pair(
                            'is %s a 2c company' % entity_name, 'no', make_context_map(entity_instance)),
                    ])

            ############################
            #      Company.stage       #
            ############################
            elif property_def.property_name == 'stage':
                stage = entity_instance.property_value_map.get('stage')
                if stage:
                    answer = 'As far as I know, %s has received %s funding from A16Z.' % (entity_name, stage.lower())
                else:
                    # keep existing answer
                    answer = qa_pairs[0].answer
                for qa_pair in qa_pairs:
                    qa_pair.answer = answer
                qa_pairs.extend([
                    entity_property_concept.new_qa_pair(question_text, answer, make_context_map(entity_instance))
                    for question_text in [
                        'current stage of %s' % entity_name,
                        'is %s funded' % entity_name,
                        'is %s seeded' % entity_name,
                        'how is %s doing' % entity_name,
                        'has %s raised any capital' % entity_name,
                        'has %s raised any money' % entity_name,
                    ]
                ])

        elif isinstance(entity_instance, Investor):
            ############################
            #      Investor.role       #
            ############################
            if property_def.property_name == 'role':
                answer = entity_instance.get_role_description()
                for qa_pair in qa_pairs:
                    qa_pair.answer = answer
                qa_pairs.extend([
                    entity_property_concept.new_qa_pair(question_text, answer, make_context_map(entity_instance))
                    for question_text in [
                        'what does %s do' % entity_name,
                        'what does %s work on' % entity_name,
                        'what is the responsibility of %s' % entity_name,
                    ]
                ])

        return qa_pairs

    def __generate_qa_pairs_about_entity_relation(self, entity_relation_concept):
        entity_instance = entity_relation_concept.entity_instance
        relation_def = entity_relation_concept.relation_def

        relation_wh_type = ConceptType.get_wh_type(relation_def.related_entity_class.entity_concept_type)
        relation_name = relation_def.relation_name
        entity_name = entity_instance.property_value_map['name']

        related_entity_value = entity_instance.relation_value_map.get(relation_name)

        if relation_def.relation_type == EntityRelation.ONE_TO_ONE:
            question_texts = [
                '%s is the %s of %s' % (relation_wh_type, relation_name, entity_name),
                'show me the %s of %s' % (relation_name, entity_name)
            ]
            #TODO
            answer = 'n/a'
        else:
            question_texts = [
                '%s are the %s of %s' % (relation_wh_type, relation_name, entity_name),
                'list all %s of %s' % (relation_name, entity_name),
                'show me all %s of %s' % (relation_name, entity_name),
                'how many %s of %s' % (relation_name, entity_name),
                'the number of %s of %s' % (relation_name, entity_name),
            ]
            if related_entity_value:
                example = ', '.join([single_intance.get_entity_id() for single_intance in related_entity_value][0:3])
                answer = 'There are %d %s in total, including %s...' % (len(related_entity_value), relation_name, example)
            else:
                answer = "Sorry, there doesn't seem to be any."

        qa_pairs = [
            entity_relation_concept.new_qa_pair(question_text, answer, make_context_map(entity_instance))
            for question_text in question_texts
        ]

        # additional stuff

        return qa_pairs


# Type-specific generators

class JobQaGenerator(object):

    def generate_qa_pairs_for_entity_class(self, entity_class):
        #TODO
        return []

    def generate_qa_pairs_about_one_property(self, entity_instance, property_def):
        qa_pairs = super(JobQaGenerator, self).generate_qa_pairs_about_one_property(entity_instance, property_def)
        entity_name = entity_instance.property_value_map['name']

        if property_def.property_name == 'name':
            qa_pairs.extend([
                QAPair('what is the position for %s' % entity_name, 'n/a', make_context_map(entity_instance)),
                QAPair('what do I work on for %s' % entity_name, 'n/a', make_context_map(entity_instance)),
                QAPair('what expertise do I need for %s' % entity_name, 'n/a', make_context_map(entity_instance)),
                QAPair('what is the requirement for %s' % entity_name, 'n/a', make_context_map(entity_instance)),
            ])

        elif property_def.property_name == 'location':
            qa_pairs.extend([
                QAPair('where is the office for %s' % entity_name, 'n/a', make_context_map(entity_instance)),
                QAPair('where do I need to work for %s' % entity_name, 'n/a', make_context_map(entity_instance)),
            ])

        return qa_pairs

    def generate_qa_pairs_about_one_relation(self, entity_instance, relation_def):
        qa_pairs = super(JobQaGenerator, self).generate_qa_pairs_about_one_relation(entity_instance, relation_def)
        entity_name = self.property_value_map['name']

        if relation_def.relation_name == 'company':
            qa_pairs.extend([
                QAPair('who is the employer of %s' % entity_name, 'n/a', make_context_map(entity_instance)),
                QAPair('which company is %s for' % entity_name, 'n/a', make_context_map(entity_instance)),
            ])

        return qa_pairs

