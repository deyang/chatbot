import json

from chatterbot import ChatBot
from wit import message
import re
import os
from utils.util import get_logger

__author__ = 'Deyang'

bot_logger = get_logger('bot')
wit_token = 'QUCDCX7MQX4FLYGONBEYLGDHKSTIUFTQ'


# identity_intent = [conversation[0] for conversation in identity_data]
# ask_company_intent = [conversation[0] for conversation in ask_company_data]
# ask_customer_intent = [conversation[0] for conversation in ask_customer_data]
# ask_doc_intent = [conversation[0] for conversation in ask_doc_data]
# ask_price_intent = [conversation[0] for conversation in ask_price_data]
# ask_product_intent = [conversation[0] for conversation in ask_product_data]
# ask_story_intent = [conversation[0] for conversation in ask_story_data]


GREETING_INTENT = 'greetings'
IDENTITY_INTENT = 'identity'
TRY_INTENT = 'try'
EMAIL_INTENT = 'email'
INSULT_AND_SEX_INTENT = 'insult_and_sex'
ASK_PRICE_INTENT = 'ask_price'
ASK_CUSTOMER_INTENT = 'ask_customer'
ASK_TEAM_INTENT = 'ask_team'
ASK_LAUNCH_INTENT = 'ask_launch'
ASK_BETA_INTENT = 'ask_beta'
ASK_WORK_CASE_INTENT = 'ask_work_case'
ASK_INTEGRATION_INTENT = 'ask_integration'
ASK_INTENT_PATTERN = re.compile('ask.*')


def load_data(file_name):
    dirpath = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(dirpath, 'data', file_name)
    fd = open(filepath, 'r')
    lines = fd.readlines()
    skipped_lines = [l for l in lines if not l.strip().startswith('#')]
    content = ''.join(skipped_lines)
    return json.loads(content)


class DummyEmailBot(object):
    def get_response(self, input):
        return "Thanks! Please sit tight. I've notified a member of the team"


class DummyFallbackBot(object):
    def get_response(self, input):
        return "I'm sorry but I don't understand what you meant by that. Right now I can only answer questions about Buddy AI. Is this somthing that I can help with?"


class EntityBot(object):
    def get_response(self, entity):
        raise NotImplementedError


class WorkCaseBot(EntityBot):
    def get_response(self, business):
        return "Yes, buddy works perfectly for %s. There is a template bot for each vertical, trained with the related domain knowledge." \
               " After training using your specific customer support data, the bot can be an expert of your business." \
               " I'm very confident of my peer bots!" % business


class IntegratePlatformBot(EntityBot):
    def get_response(self, platform):
        return "Yes, buddy bot can be integrated with %s. " \
               "We currently support integrations with Kik, Telegram, Twitter DM, " \
               "InterComm, Smooch, Slack, HipChat, Twilio, Zendesk and Desk.com. Facebook messenger and many more are coming!" % platform


class Bots(object):

    def _install_bot(self, intent, data):
        new_bot = self.get_new_bot("%s_bot" % intent)
        for conversation in data:
            new_bot.train(conversation)
        self.set_readonly(new_bot)
        self.intent_to_bot_dict[intent] = new_bot

    def __init__(self):
        self.intent_to_bot_dict = dict()
        # load data
        identity_data = load_data('identity.qa')
        ask_company_data = load_data('company.qa')
        ask_customer_data = load_data('customer.qa')
        ask_doc_data = load_data('doc.qa')
        ask_price_data = load_data('price.qa')
        ask_product_data = load_data('product.qa')
        ask_story_data = load_data('story.qa')
        ask_beta_data = load_data('beta.qa')
        ask_integration_data = load_data('integration.qa')
        ask_team_data = load_data('team.qa')
        ask_launch_data = load_data('launch.qa')
        additional_greeting = load_data('greeting.qa')
        try_data = load_data('try.qa')
        insult_and_sex_data = load_data('insult_and_sex.qa')

        self.general_bot = self.get_new_bot('general_bot')
        self.general_bot.train(
            "chatterbot.corpus.english",
        )
    
        greeting_bot = self.get_new_bot('greeting_bot')
        greeting_bot.train("chatterbot.corpus.english.greetings")
        for conversation in additional_greeting:
            greeting_bot.train(conversation)
        self.set_readonly(greeting_bot)
        self.intent_to_bot_dict[GREETING_INTENT] = greeting_bot

        self._install_bot(IDENTITY_INTENT, identity_data)
        self._install_bot(TRY_INTENT, try_data)
        self._install_bot(INSULT_AND_SEX_INTENT, insult_and_sex_data)
        self._install_bot(ASK_PRICE_INTENT, ask_price_data)
        self._install_bot(ASK_CUSTOMER_INTENT, ask_customer_data)
        self._install_bot(ASK_TEAM_INTENT, ask_team_data)
        self._install_bot(ASK_LAUNCH_INTENT, ask_launch_data)
        self._install_bot(ASK_BETA_INTENT, ask_beta_data)

        self.company_bot = self.get_new_bot('company_bot')
        all_data = ask_company_data + ask_doc_data + ask_story_data + \
            ask_product_data + ask_integration_data
        for conversation in all_data:
            self.company_bot.train(conversation)
        self.set_readonly(self.company_bot)

        self.email_bot = DummyEmailBot()
        self.fall_back_bot = DummyFallbackBot()
        self.work_case_bot = WorkCaseBot()
        self.integrate_platform_bot = IntegratePlatformBot()

    @staticmethod
    def get_new_bot(name):
        return ChatBot(
            name,
            logic_adapters=[
                "chatterbot.adapters.logic.ClosestMatchAdapter",
                "chatterbot.adapters.logic.ClosestMeaningAdapter"
            ],
            io_adapters=[
                "chatterbot.adapters.io.NoOutputAdapter"
            ],
            database=name + "_database.db"
        )

    @staticmethod
    def set_readonly(bot):
        for adapter in bot.storage_adapters:
            adapter.read_only = True

    def chat(self, in_msg):
        entity = None
        # hack based on html. No need to query wit.ai
        if 'mailto:' in in_msg:
            intent = EMAIL_INTENT
            confidence = 1.0
        else:
            resp = message(wit_token, in_msg)
            try:
                intent = resp['outcomes'][0]['intent']
                confidence = resp['outcomes'][0]['confidence']
                bot_logger.info("Predicting intent: %s" % intent)
                bot_logger.info("Predicting confidence: %s" % confidence)
            except Exception as e:
                bot_logger.error("Wit exception: %s" % e)
                bot_logger.error("Wit reps: %s" % resp)

        if confidence < 0.5:
            bot = self.fall_back_bot
        # handle entity intent first
        elif intent == ASK_WORK_CASE_INTENT and len(resp['outcomes'][0]['entities']) > 0:
            entity = resp['outcomes'][0]['entities']['business'][0]['value']
            bot = self.work_case_bot
        elif intent == ASK_INTEGRATION_INTENT and len(resp['outcomes'][0]['entities']) > 0:
            entity = resp['outcomes'][0]['entities']['platform'][0]['value']
            bot = self.integrate_platform_bot
        elif intent in self.intent_to_bot_dict:
            bot = self.intent_to_bot_dict[intent]
        elif ASK_INTENT_PATTERN.match(intent):
            bot = self.company_bot
        else:
            bot = self.general_bot

        if entity:
            answer = bot.get_response(entity)
        else:
            answer = bot.get_response(in_msg)
        bot_logger.info("In msg: <%s> and out msg: <%s>" % (in_msg, answer))
        return answer

if __name__ == '__main__':
    bots = Bots()
    while True:
        in_msg = raw_input()
        out_msg = bots.chat(in_msg)
        print "Buddy bot :> %s" % out_msg
