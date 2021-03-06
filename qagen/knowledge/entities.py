from qagen.qa.utils import construct_job_search_url


class ConceptType(object):

    THING = 1,
    PERSON = 2,
    URL = 3,

    __wh_type_map = {
        THING: 'what',
        PERSON: 'who',
        URL: 'where'
    }

    @classmethod
    def get_wh_type(cls, concept_type):
        return cls.__wh_type_map[concept_type]


class EntityProperty(object):

    def __init__(self, property_name, concept_type, is_hidden=False):
        self.property_name = property_name
        self.concept_type = concept_type
        self.is_hidden = is_hidden


class EntityRelation(object):

    ONE_TO_ONE = 1
    ONE_TO_MANY = 2

    def __init__(self, relation_name, related_entity_class, relation_type):
        self.relation_name = relation_name
        self.related_entity_class = related_entity_class
        self.relation_type = relation_type


class BaseEntity(object):

    entity_concept_type = ConceptType.THING
    property_def_map = {}
    relation_def_map = {}

    def __init__(self, property_value_map):
        # check the required metadata
        if not hasattr(self, 'entity_concept_type'):
            raise Exception("entity_concept_type is not specified")
        if not hasattr(self, 'property_def_map'):
            raise Exception("property_def_map is not specified")
        if not hasattr(self, 'relation_def_map'):
            raise Exception("relation_def_map is not specified")

        for key in property_value_map:
            if key not in self.__class__.property_def_map:
                raise Exception(key + ' is not found in the property definition map in ' + self.__class__.__name__)

        # copy over all the input variables as property value
        self.property_value_map = dict(property_value_map)
        self.relation_value_map = {}

    def __repr__(self):
        entity_type = self.__class__.__name__
        entity_name = self.get_entity_id()
        return '%s[%s]' % (entity_type, entity_name)

    def get_entity_id(self):
        """by default use the name property as identifier among all entity instances
        subclass may use different property as id by overriding this method"""
        return self.property_value_map['name']

    def get_entity_self_description(self):
        """used for answering questions about the entity itself. subclass may override implementation"""
        return '%s is a %s' % (self.get_entity_id(), self.__class__.__name__)


# Entity types and definitions

class Company(BaseEntity):

    def get_entity_self_description(self):
        return '%s is a company invested by Andreessen Horowitz. You can ask me more about its location, founders, ' \
               'type of business, etc.' % self.property_value_map['name']

    def add_job(self, job_entity):
        jobs = self.relation_value_map.get('job openings')
        if not jobs:
            jobs = self.relation_value_map['job openings'] = []
        jobs.append(job_entity)

    def get_job_search_url(self):
        return construct_job_search_url(company_id=self.property_value_map['company_id'])


class Job(BaseEntity):

    def get_entity_id(self):
        return 'JOBOP' + self.property_value_map['job_id']

    def get_entity_self_description(self):
        return 'This is a job opening for %s at %s. You can find more details about it at %s' %(
            self.property_value_map['title'],
            self.property_value_map['company name'],
            self.get_job_detail_page_url()
        )

    def get_job_detail_page_url(self):
        return 'http://portfoliojobs.a16z.com/jobdetail.php?jobid=%s' % self.property_value_map['job_id']


class Investor(BaseEntity):

    def get_entity_self_description(self):
        return self.get_role_description() + \
               ' You can ask me more to show the LinkedIn profile, the picture, etc.'

    def get_role_description(self):
        role_description_raw = self.property_value_map['role']
        title = self.__extract_single_term_title(role_description_raw)
        if title:
            return '%s is a %s at Andreessen Horowitz.' % (self.property_value_map['name'], title)
        else:
            return '%s works at Andreessen Horowitz and is in charge of %s.' \
                   % (self.property_value_map['name'], role_description_raw.replace(':', ','))


    @staticmethod
    def __extract_single_term_title(role_description):
        role_first_part = role_description.split(':')[0]
        for hot_word in ['advisor', 'partner', 'professor']:
            if hot_word in role_first_part:
                hot_word_plural = hot_word + 's'
                if hot_word_plural in role_first_part:
                    return role_first_part.replace(hot_word_plural, hot_word)
                else:
                    return role_first_part
        return None


class A16Z(Company):

    def get_entity_self_description(self):
        return 'Andreessen Horowitz (A16Z) is a Silicon Valley-based venture capital firm. ' \
               'You can ask me more about its team, portfolio, contact info, etc.'

    def get_all_2b_companies(self):
        return [company
                for company in self.relation_value_map['portfolio companies']
                if company.property_value_map['business model'] == 'to enterprise']

    def get_all_2c_companies(self):
        return [company
                for company in self.relation_value_map['portfolio companies']
                if company.property_value_map['business model'] == 'to consumer']

    def get_all_seed_companies(self):
        return [company
                for company in self.relation_value_map['portfolio companies']
                if company.property_value_map['stage'] == 'seed']

    def get_all_venture_companies(self):
        return [company
                for company in self.relation_value_map['portfolio companies']
                if company.property_value_map['stage'] == 'venture']
    
    def get_all_board_partners(self):
        return [investor
                for investor in self.relation_value_map['people']
                if 'board partner' in investor.property_value_map['role'].lower()]


Company.entity_concept_type = ConceptType.THING
Company.property_def_map = {
    'name': EntityProperty('name', ConceptType.THING),
    'founder': EntityProperty('founder', ConceptType.PERSON),
    'location': EntityProperty('location', ConceptType.THING),
    'website': EntityProperty('website', ConceptType.URL),
    'type of business': EntityProperty('type of business', ConceptType.THING),
    'business model': EntityProperty('business model', ConceptType.THING),
    'stage': EntityProperty('stage', ConceptType.THING),
    # company_id is only for answer generation, hence hidden
    'company_id': EntityProperty('company_id', ConceptType.THING, is_hidden=True)
}
Company.relation_def_map = {
    'job openings': EntityRelation('job openings', Job, EntityRelation.ONE_TO_MANY),
    'example of job opening': EntityRelation('example of job opening', Job, EntityRelation.ONE_TO_ONE)
}


Job.entity_concept_type = ConceptType.THING
Job.property_def_map = {
    'title': EntityProperty('title', ConceptType.THING),
    'location': EntityProperty('location', ConceptType.THING),

    # we don't really care to ask about the function, but we need this property
    # in order to come up with class-level queries
    'function': EntityProperty('function', ConceptType.THING, is_hidden=True),
    # it's possible that the company name cannot be associated back to a company entity
    # in this case we save the name of the company as a normal property so that it's able
    # to handle some basic question
    'company name': EntityProperty('company name', ConceptType.THING, is_hidden=True),
    # these ids are only for answer generation, hence hidden
    'job_id': EntityProperty('job_id', ConceptType.THING, is_hidden=True),
    'function_id': EntityProperty('function_id', ConceptType.THING, is_hidden=True),
    'location_id': EntityProperty('location_id', ConceptType.THING, is_hidden=True),
    'company_id': EntityProperty('company_id', ConceptType.THING, is_hidden=True)
}
Job.relation_def_map = {
    'company': EntityRelation('company', Company, EntityRelation.ONE_TO_ONE)
}


Investor.entity_concept_type = ConceptType.PERSON
Investor.property_def_map = {
    'name': EntityProperty('name', ConceptType.THING),
    'role': EntityProperty('role', ConceptType.THING),
    'picture': EntityProperty('picture', ConceptType.URL),
    'profile': EntityProperty('profile', ConceptType.URL),
    'linkedin': EntityProperty('linkedin', ConceptType.URL),
}
Investor.relation_def_map = {}


A16Z.entity_concept_type = Company.entity_concept_type
A16Z.property_def_map = dict(Company.property_def_map)
A16Z.relation_def_map = dict(Company.relation_def_map)
# additional properties and relations
A16Z.property_def_map['contact info'] = EntityProperty('contact info', ConceptType.URL)
A16Z.property_def_map['mailing address'] = EntityProperty('mailing address', ConceptType.THING)
A16Z.property_def_map['twitter account'] = EntityProperty('twitter account', ConceptType.THING)
A16Z.relation_def_map['portfolio companies'] = EntityRelation('portfolio companies', Company, EntityRelation.ONE_TO_MANY)
A16Z.relation_def_map['people'] = EntityRelation('people', Investor, EntityRelation.ONE_TO_MANY)
A16Z.relation_def_map['example of portfolio company'] = EntityRelation('example of portfolio company', Company, EntityRelation.ONE_TO_ONE)
A16Z.relation_def_map['example of investor'] = EntityRelation('example of investor', Investor, EntityRelation.ONE_TO_ONE)


def find_entity_class_by_name(entity_class_name):
    for known_entity_type in [Company, Job, Investor, A16Z]:
        if known_entity_type.__name__ == entity_class_name:
            return known_entity_type
    return None
