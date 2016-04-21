import urllib2
import json
import sys
reload(sys)  # Reload does the trick!
sys.setdefaultencoding('UTF8')
from bs4 import BeautifulSoup
from urlparse import urlparse

from qagen.knowledge.entities import *
from qagen.knowledge.json_converter import EntityJsonConverter, ENTITY_JSON_ID, ENTITY_JSON_RELATIONS,\
    ENTITY_JSON_RELATION_REF_ENTITY_TYPE, ENTITY_JSON_RELATION_REF_ENTITY_ID, ENTITY_JSON_RELATION_REF_ENTITY_IDS


class KnowledgeDataProvider(object):

    def __init__(self):
        self.entity_map = {
            A16Z: [],
            Company: [],
            Investor: [],
            Job: [],
        }

    def add_entity(self, entity_instance):
        self.entity_map[entity_instance.__class__].append(entity_instance)

    def get_all_entity_types(self):
        return self.entity_map.keys()

    def get_all_instances_of_type(self, entity_class):
        return self.entity_map[entity_class]

    def get_entity_instance(self, entity_class, entity_id):
        for entity_instance in self.get_all_instances_of_type(entity_class):
            if entity_instance.get_entity_id() == entity_id:
                return entity_instance
        return None


class WebCrawlerKnowledgeDataProvider(KnowledgeDataProvider):

    def __init__(self):
        super(WebCrawlerKnowledgeDataProvider, self).__init__()
        self.__crawl_all_entity_data()
        self.__reconstruct_entity_relations()

    def __crawl_all_entity_data(self):

        print 'Initializing company:A16Z...'
        a16z = A16Z({
            'name': 'Andreesen Horowitz',
            'founder': 'Marc Andreesen and Ben Horowitz',
            'location': 'Melo Park, California',
            'website': 'a16z.com',
            'type of business': 'venture capital',
            'business model': None,
            'stage': None,
            'contact info': 'http://a16z.com/about/contact/',
        })
        self.add_entity(a16z)

        print 'Crawling http://a16z.com/portfolio/...'
        print 'Companies in venture/growth stage...'
        for div in self.__find_company_data_divs_from_url('http://a16z.com/portfolio/'):
            self.add_entity(self.__parse_company_data(div))
        print 'Companies in seed stage...'
        for div in self.__find_seed_company_data_divs_from_url('http://a16z.com/portfolio/'):
            self.add_entity(self.__parse_seed_company_data(div))

        print 'Crawling http://a16z.com/about/team/'
        team_member_groups = self.__collect_team_member_profile_urls('http://a16z.com/about/team/')
        for team_member_group_info in team_member_groups:
            for investor_url in team_member_group_info['urls']:
                self.add_entity(self.__parse_investor_data(investor_url, team_member_group_info['role_description']))

    def __reconstruct_entity_relations(self):

        print 'Reconstructing entity relations...'

        the_a16z = self.get_all_instances_of_type(A16Z)[0]
        all_companies = self.get_all_instances_of_type(Company)
        all_investors = self.get_all_instances_of_type(Investor)

        print 'A16Z -> COMPANYs'
        the_a16z.relation_value_map['portfolio companies'] = all_companies
        print 'A16Z -> INVESTORs'
        the_a16z.relation_value_map['people'] = all_investors
        # a16z -> POSTs
        # a16z -> PODCASTs

        # COMPANY -> JOBs
        # JOB -> a COMPANY

    @staticmethod
    def __find_company_data_divs_from_url(url):
        page = urllib2.urlopen(url)
        soup = BeautifulSoup(page, 'html.parser')
        return soup.findAll('div', 'company')

    @staticmethod
    def __parse_company_data(div):
        name = div.select_one('.meta > li:nth-of-type(1) > h2').next_sibling.strip()
        founder = div.select_one('.meta > li:nth-of-type(2) > h2').next_sibling.strip()
        location = div.select_one('.meta > li:nth-of-type(3) > h2').next_sibling.strip()
        website = div.select_one('.meta > li:nth-of-type(4) > a')['href'].strip()
        type_of_business = div.select_one('.meta > li:nth-of-type(5) > h2').next_sibling.strip()
        business_model = 'to consumer' if 'consumer' in div['class'] else 'to enterprise'

        return Company({
            'name': name,
            'founder': founder,
            'location': location,
            'website': website,
            'type of business': type_of_business,
            'business model': business_model,
            'stage': 'venture',
        })

    @staticmethod
    def __find_seed_company_data_divs_from_url(url):
        page = urllib2.urlopen(url)
        soup = BeautifulSoup(page, 'html.parser')
        return soup.findAll('div', 'company-seed')

    @staticmethod
    def __parse_seed_company_data(div):
        website = div.select_one('.img-wrapper')['href'].strip()

        return Company({
            'name': WebCrawlerKnowledgeDataProvider.__guess_company_name_by_website(website),
            'founder': None,
            'location': None,
            'website': website,
            'type of business': None,
            'business model': None,
            'stage': 'seed',
        })

    @staticmethod
    def __guess_company_name_by_website(url):
        domain_name_no_suffix = urlparse(url).netloc.split('.')[-2]
        terms = domain_name_no_suffix.split('-')
        terms_capitalized = [term.capitalize() for term in terms]
        return ' '.join(terms_capitalized)

    @staticmethod
    def __collect_team_member_profile_urls(url):
        """returns list of urls grouped by role of investor"""
        page = urllib2.urlopen(url)
        soup = BeautifulSoup(page, 'html.parser')

        result = []
        for team_section_div in soup.select('#main .team-section'):
            role_description = WebCrawlerKnowledgeDataProvider.__parse_role_description(team_section_div)
            urls = WebCrawlerKnowledgeDataProvider.__parse_team_member_profile_urls(team_section_div)
            result.append({
                'role_description': role_description,
                'urls': urls
            })

        return result

    @staticmethod
    def __parse_role_description(div):
        text = div.select_one('.team-heading').get_text().strip().lower()
        if text.endswith('.'):
            text = text[0:-1]
        return text

    @staticmethod
    def __parse_team_member_profile_urls(div):
        return [a_tag['href'] for a_tag in div.select('.team-member > a')]

    @staticmethod
    def __parse_investor_data(page_url, role_description):
        print 'Crawling %s...' % page_url

        page = urllib2.urlopen(page_url)
        soup = BeautifulSoup(page, 'html.parser')

        investor_data_div = soup.select_one('.team-member-info')
        name = investor_data_div.select_one('.team-member-name').get_text()
        photo_url = investor_data_div.select_one('img.team-member-photo')['src']
        linkedin_url = None
        for a_tag_in_bio in investor_data_div.select('.team-member-bio a[href]'):
            url = a_tag_in_bio['href']
            if 'linkedin.com' in url:
                linkedin_url = url
                print 'Found LinkedIn profile: ' + linkedin_url

        return Investor({
            'name': name,
            'role': role_description,
            'picture': photo_url,
            'profile': page_url,
            'linkedin': linkedin_url,
        })


class JsonFileKnowledgeDataProvider(KnowledgeDataProvider):

    def __init__(self, file_path):
        super(JsonFileKnowledgeDataProvider, self).__init__()
        with open(file_path, 'r') as input_file:
            json_data = json.load(input_file)
            # unmarshall the entity properties before reconstructing relations
            self.__load_entity_property_data_from_dict(json_data)
            self.__load_entity_relations_from_dict(json_data)

    def __load_entity_property_data_from_dict(self, json_dict):
        for known_entity_type in self.entity_map:
            entity_json_list = json_dict.get(known_entity_type.__name__)
            if entity_json_list:
                for entity_data in entity_json_list:
                    self.add_entity(EntityJsonConverter.load_from_json_dict(entity_data))

    def __load_entity_relations_from_dict(self, json_dict):
        for known_entity_type in self.entity_map:
            for entity_data in json_dict.get(known_entity_type.__name__):
                # search the already-materialized instance
                entity_id = entity_data[ENTITY_JSON_ID]
                entity_instance = self.get_entity_instance(known_entity_type, entity_id)

                for relation_name, relation_ref_dat in entity_data[ENTITY_JSON_RELATIONS].iteritems():
                    related_entity_class_name = relation_ref_dat[ENTITY_JSON_RELATION_REF_ENTITY_TYPE]
                    related_entity_class = find_entity_class_by_name(related_entity_class_name)
                    related_entity_id = relation_ref_dat.get(ENTITY_JSON_RELATION_REF_ENTITY_ID)
                    related_entity_ids = relation_ref_dat.get(ENTITY_JSON_RELATION_REF_ENTITY_IDS)

                    # search the related entities
                    if related_entity_id:
                        entity_instance.relation_value_map[relation_name] = \
                            self.get_entity_instance(related_entity_class, related_entity_id)
                    elif related_entity_ids:
                        entity_instance.relation_value_map[relation_name] = \
                            [self.get_entity_instance(related_entity_class, single_id) for single_id in related_entity_ids]
                    else:
                        raise Exception('Neither %s or %s is found in the relation reference'
                                        % (ENTITY_JSON_RELATION_REF_ENTITY_ID, ENTITY_JSON_RELATION_REF_ENTITY_IDS))


