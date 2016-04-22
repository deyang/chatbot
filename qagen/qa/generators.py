from qagen.knowledge.entities import *
from qagen.qa.utils import make_context_map
from qagen.qa.qa_pair import EntityClassLevelQA, EntityInstanceSelfQA, EntityPropertyQA, EntityRelationQA


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
        entity_class = entity_class_concept.entity_class

        if entity_class == A16Z:
            the_a16z = self.data_provider.get_all_instances_of_type(A16Z)[0]
            ##########################
            #      A16Z.class        #
            ##########################
            qa_pairs = [
                entity_class_concept.new_qa_pair(
                    'what does A16Z mean',
                    "A16Z is the abbreviation for Andreessen Horowitz, the combination of the two founder's last names.",
                    make_context_map(the_a16z)
                ),
                entity_class_concept.new_qa_pair(
                    'where did you get all the data about A16Z',
                    'All the knowledge that I know of are crawled from the website a16z.com',
                    make_context_map()
                ),
                entity_class_concept.new_qa_pair(
                    'who are you',
                    "I am robot that answers questions about A16Z. "
                    "So you'd better not expect me to know anything that is not available on the a16z.com website",
                    make_context_map()
                ),
                entity_class_concept.new_qa_pair(
                    'what kind of question can I ask about A16Z',
                    'You can ask me about its team, portfolio, job openings, contact info, etc. '
                    'Basically anything that is available on the a16z.com website',
                    make_context_map()
                ),
            ]

        elif entity_class == Company:
            ##########################
            #     Company.class      #
            ##########################
            qa_pairs = [
                entity_class_concept.new_qa_pair(
                    'what kind of question can I ask about a company',
                    'You can ask me about its name, founder, location, type of business and even job openings. '
                    'Basically anything that is available on the a16z.com website',
                    make_context_map()
                ),
            ]

        elif entity_class == Investor:
            ##########################
            #     Investor.class     #
            ##########################
            qa_pairs = [
                entity_class_concept.new_qa_pair(
                    'what kind of question can I ask about an investor',
                    'You can ask me about his/her name, role, picture, linkedin profile, etc. '
                    'Basically anything that is available on the a16z.com website',
                    make_context_map()
                ),
            ]

        elif entity_class == Job:
            ##########################
            #        Job.class       #
            ##########################
            qa_pairs = [
                entity_class_concept.new_qa_pair(
                    'what kind of question can I ask about job openings',
                    'You can ask me about its company, title, location, etc. '
                    'Basically anything that is available on the a16z.com website',
                    make_context_map()
                ),
            ]

        return qa_pairs

    def __generate_qa_pairs_about_entity_instance(self, entity_self_concept):
        entity_instance = entity_self_concept.entity_instance
        wh_type = ConceptType.get_wh_type(entity_instance.__class__.entity_concept_type)
        entity_name = entity_instance.get_entity_id()

        qa_pairs = [
            entity_self_concept.new_qa_pair(question_text, entity_instance.get_entity_self_description(),
                                            make_context_map(entity_instance))
            for question_text in [
                '%s is %s' % (wh_type, entity_name),
                'information about %s' % entity_name,
                'tell me about %s' % entity_name,
                'show me %s' % entity_name,
                'I want to know about %s' % entity_name,
                'do you know about %s' % entity_name,
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

        ##########################
        #      Job.self          #
        ##########################
        elif isinstance(entity_instance, Job):
            # NOTE: we remove some of the common form of the self question that doesn't make sense for a Job
            qa_pairs = [
                entity_self_concept.new_qa_pair(question_text, entity_instance.get_entity_self_description(),
                                                make_context_map(entity_instance))
                for question_text in [
                    'tell me more about %s' % entity_name,
                    'I want to know about %s' % entity_name
                ]
            ]
            apply_answer = 'You can apply anytime online by visiting %s' % entity_instance.get_job_detail_page_url()
            qa_pairs.extend([
                entity_self_concept.new_qa_pair(question_text, apply_answer, make_context_map(entity_instance))
                for question_text in [
                    'how can I apply for %s' % entity_name,
                    'I want to apply for %s' % entity_name,
                ]
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
        entity_name = entity_instance.get_entity_id()

        property_value = entity_instance.property_value_map.get(property_name)
        if property_value:
            if property_def.concept_type == ConceptType.URL:
                answer = 'You can find the %s of %s at %s' % (property_name, entity_name, property_value)
            elif property_def.property_name == 'name':
                answer = property_value
            else:
                answer = 'The %s of %s is %s.' % (property_name, entity_name, property_value)
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

        if isinstance(entity_instance, A16Z):
            ##########################
            #    A16Z.contact info   #
            ##########################
            if property_def.property_name == 'contact info':
                answer = qa_pairs[0].answer # use existing answer
                qa_pairs.extend([
                    entity_property_concept.new_qa_pair(question_text, answer, make_context_map(entity_instance))
                    for question_text in [
                        'how do I contact %s' % entity_name,
                        'I want to talk to someone at %s' % entity_name,
                        'how can I talk to someone at %s' % entity_name,
                        'I want to talk to an investor at %s' % entity_name,
                        'how can I get invested by %s' % entity_name,
                        ]
                ])

        elif isinstance(entity_instance, Company):
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

        elif isinstance(entity_instance, Job):
            answer = 'The %s of this job opening is for %s' % (property_name, property_value)
            for qa_pair in qa_pairs:
                qa_pair.answer = answer
            ############################
            #      Job.title           #
            ############################
            if property_def.property_name == 'title':
                qa_pairs.extend([
                    entity_property_concept.new_qa_pair(question_text, answer, make_context_map(entity_instance))
                    for question_text in [
                        'what is the position for %s' % entity_name,
                        'what do I work on for %s' % entity_name,
                        'what expertise do I need for %s' % entity_name,
                        'what is the requirement for %s' % entity_name,
                    ]
                ])

            ############################
            #      Job.location        #
            ############################
            elif property_def.property_name == 'location':
                qa_pairs.extend([
                    entity_property_concept.new_qa_pair(question_text, answer, make_context_map(entity_instance))
                    for question_text in [
                        'where is the office for %s' % entity_name,
                        'where do I need to work for %s' % entity_name,
                    ]
                ])

        return qa_pairs

    def __generate_qa_pairs_about_entity_relation(self, entity_relation_concept):
        entity_instance = entity_relation_concept.entity_instance
        relation_def = entity_relation_concept.relation_def

        relation_wh_type = ConceptType.get_wh_type(relation_def.related_entity_class.entity_concept_type)
        relation_name = relation_def.relation_name
        entity_name = entity_instance.get_entity_id()

        related_entity_value = entity_instance.relation_value_map.get(relation_name)

        if relation_def.relation_type == EntityRelation.ONE_TO_ONE:
            question_texts = [
                '%s is the %s of %s' % (relation_wh_type, relation_name, entity_name),
                'show me the %s of %s' % (relation_name, entity_name)
            ]
            if related_entity_value:
                answer = 'The %s of %s is %s.' % (relation_name, entity_name, related_entity_value.get_entity_id())
                # add the related entity to the context, with higher precedence
                context = make_context_map(entity_instance, related_entity_value)
            else:
                answer = 'Sorry, I cannot find the information about the %s of %s.' % (relation_name, entity_name)
                context = make_context_map(entity_instance)
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
            context = make_context_map(entity_instance)

        qa_pairs = [
            entity_relation_concept.new_qa_pair(question_text, answer, context)
            for question_text in question_texts
        ]

        # additional stuff

        if isinstance(entity_instance, A16Z):
            ############################
            #  A16Z.portfolio company  #
            ############################
            if relation_def.relation_name == 'portfolio companies':
                answer = qa_pairs[0].answer
                context = qa_pairs[0].context_list
                qa_pairs.extend([
                    entity_relation_concept.new_qa_pair(question_text, answer, context)
                    for question_text in [
                        'who are invested by %s' % entity_name,
                        'companies invested by %s' % entity_name,
                        'who has %s invested in' % entity_name,
                        'show all the investments made by %s' % entity_name,
                    ]
                ])

            ############################
            #       A16Z.people        #
            ############################
            elif relation_def.relation_name == 'people':
                # use existing answer
                answer = qa_pairs[0].answer
                context = qa_pairs[0].context_list
                qa_pairs.extend([
                    entity_relation_concept.new_qa_pair(question_text, answer, context)
                    for question_text in [
                        'who work for %s' % entity_name,
                        'show me the team of %s' % entity_name,
                        'show me the folks at %s' % entity_name,
                    ]
                ])
                # specific question on board partner
                board_partners = entity_instance.get_all_board_partners()
                example_names = ', '.join([partner.get_entity_id() for partner in board_partners][0:3])
                qa_pairs.extend([
                    entity_relation_concept.new_qa_pair(
                        'who are board partners at %s' % entity_name,
                        'I found %d board partners at %s, including %s...'
                            % (len(board_partners), entity_name, example_names),
                        context
                    )
                ])

            ################################################
            #       A16Z.example portfolio company         #
            ################################################
            elif relation_def.relation_name == 'example portfolio company':
                # tweak answer
                answer = 'For example, ' + related_entity_value.get_entity_self_description()
                for qa_pair in qa_pairs:
                    qa_pair.answer = answer

            #######################################
            #       A16Z.example investor         #
            #######################################
            elif relation_def.relation_name == 'example investor':
                # tweak answer
                answer = 'For example, ' + related_entity_value.get_entity_self_description()
                for qa_pair in qa_pairs:
                    qa_pair.answer = answer

        elif isinstance(entity_instance, Company):
            ############################
            #  Company.job openings    #
            ############################
            if relation_def.relation_name == 'job openings':
                if related_entity_value:
                    answer = 'I found %d jobs available at %s. You can see the full list at %s' \
                             % (len(related_entity_value), entity_name, entity_instance.get_job_search_url())
                else:
                    answer = "Sorry, I cannot find any job available at %s." % entity_name
                for qa_pair in qa_pairs:
                    qa_pair.answer = answer
                qa_pairs.extend([
                    entity_relation_concept.new_qa_pair(question_text, answer, make_context_map(entity_instance))
                    for question_text in [
                        'jobs at %s' % entity_name,
                        'I want to work for %s' % entity_name,
                        'how can I work for %s' % entity_name,
                    ]
                ])

            #######################################
            #     Company.example job opening     #
            #######################################
            elif relation_def.relation_name == 'example job opening':
                if related_entity_value:
                    # tweak answer
                    answer = 'I found a job opening for %s at %s. For more results like this, you can visit %s' \
                             % (related_entity_value.property_value_map['title'], entity_name,
                                entity_instance.get_job_search_url())
                    for qa_pair in qa_pairs:
                        qa_pair.answer = answer

        elif isinstance(entity_instance, Job):
            ############################
            #      Job.company         #
            ############################
            if relation_def.relation_name == 'company':
                if related_entity_value:
                    answer = 'The company of this job opening is %s.' % related_entity_value.get_entity_id()
                    context = make_context_map(entity_instance, related_entity_value)
                else:
                    answer = 'This job opening is for %s, ' \
                             'but unfortunately I cannot find any more information about this company' \
                             % entity_instance.property_value_map['company name']
                    context = make_context_map(entity_instance)
                for qa_pair in qa_pairs:
                    qa_pair.answer = answer
                qa_pairs.extend([
                    entity_relation_concept.new_qa_pair(question_text, answer, context)
                    for question_text in [
                        'which company is %s for' % entity_name,
                    ]
                ])

        return qa_pairs
