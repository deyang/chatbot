__author__ = 'Deyang'
from chatterbot import ChatBot
from wit import message
import re

wit_token = 'QUCDCX7MQX4FLYGONBEYLGDHKSTIUFTQ'


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

identity_data = [
    [
        "What's your name?",
        "My name is Buddy Bot."
    ],
    [
        "Who are you?",
        "I'm Buddy AI bot."
    ],
    [
        "What can you do?",
        "I can answer your questions. Try to ask me about Buddy AI."
    ],
    [
        "What can you do for me?",
        "I can answer your questions. Try to ask me about Buddy AI."
    ],
    [
        "How can you help me?",
        "I can answer your questions. Try to ask me about Buddy AI."
    ],
    [
        "Are you a human?",
        "No, I'm a chatbot."
    ],
    [
        "Are you a chatbot?",
        "Yes, you are right."
    ],
    [
        "Are you a male?",
        "I guess not. I am a chatbot"
    ],
    [
        "Are you a female?",
        "I guess not. I am a chatbot"
    ],
    [
        "How do you work?",
        "I can learn from conversations. The more I talk, the more intelligent I am"
    ],
    [
        "What else can you do?",
        "My primary job is to answer customer questions. I can chat with people :)"
    ],
    [
        "Who created you?",
        "I'm developed by Buddy AI"
    ],
    [
        "Who developed you?",
        "I was developed by Buddy AI"
    ],
    [
        "Is there a human behind you?",
        "No, I'm a fully automated chatbot. However, I can escalate to a team member when necessary."
    ],
    [
        "How do you learn?",
        "I learn from conversations and literally any text data"
    ]
]

identity_intent = [conversation[0] for conversation in identity_data]


ask_company_data = [
    [
        "What is Buddy AI?",
        "Buddy AI is a startup building AI technologies for modernized customer communication."
    ],
    [
        "Tell me more about Buddy",
        "Sure. Buddy AI is a startup building AI technologies for modernized customer communication."
    ],
    [
        "Tell me more about your company",
        "Sure. Buddy AI is a startup building AI technologies for modernized customer communication."
    ],
    [
        "What does your company do?",
        "Sure. Buddy AI is a startup building AI technologies for modernized customer communication."
    ],
    [
        "How can I start?",
        "Leave me an email and I will ping our team. We'll get back to you within one hour."
    ],
    [
        "How does Buddy work?",
        "Buddy AI takes any ubiquitous data (forum, customer emails, dev docs, FAQs, internal knowledge base, etc), and then I will learn from the data and provide fully-automated or suggested answers to customer questions."
    ],
    [
        "Why do I need Buddy?",
        "I can provide 24/7 real-time customer self-service and also assist a human customer support agent. There's almost zero effort to set me up. Just feed me whatever data you have."
    ],
    [
        "How can Buddy AI help me?",
        "I can provide 24/7 real-time customer self-service and also assist a human customer support agent. There's almost zero effort to set me up. Just feed me whatever data you have."
    ],
]

ask_customer_data = [
    [
        "Who are your customers?",
        "We are opening a beta program at present and we work very closely with these companies.  I can't release them over a chat, but if you could give me some more details I'd be happy to get some further information over to you."
    ],
    [
        "Who is using Buddy now?",
        "We are opening a beta program at present and we work very closely with these companies.  I can't release them over a chat, but if you could give me some more details I'd be happy to get some further information over to you."
    ],
]

ask_doc_data = [
    [
        "Can you show me some documentations?",
        "Sure. You can learn more at everbuddy.io"
    ],
    [
        "Do you have an API?",
        "We have a full REST API. If you are interested in learning more, just respond and I'll get one of the team"
    ]
]

ask_price_data = [
    [
        "How does it charge?",
        "Buddy AI is currently in private beta and pricing is dependent on multiple variables. To learn more about our beta program, drop us an e-mail to info@everbuddy.io"
    ],
    [
        "How much is it?",
        "Buddy AI is currently in private beta and pricing is dependent on multiple variables. To learn more about our beta program, drop us an e-mail to info@everbuddy.io"
    ],
    [
        "What is the price model?",
        "Buddy AI is currently in private beta and pricing is dependent on multiple variables. To learn more about our beta program, drop us an e-mail to info@everbuddy.io"
    ]
]

ask_product_data = [
    [
        "What are the technologies used here?",
        "I'm using Natural Language Processing to understand your questions and machine learning to find the best answer"
    ],
    [
        "What are the features?",
        "Buddy AI takes any ubiquitous data (forum, customer emails, dev docs, FAQs, internal knowledge base, etc), and then I will learn from the data and provide fully-automated or suggested answers to customer questions."
    ],
]

ask_story_data = [
    [
        'Could you give me some examples?',
        'Sure. For example, I can learn from FAQs and then I will be able to answer customer questions 24/7 online.'
    ],
    [
        "Could you tell me an example of how you work?",
        "Sure. I can serve as a customer service chatbot, meaning I will answer customer questions 24/7 online. Or I can provide suggested answer to a human agent."
    ],
    [
        "What are the successful stories?",
        "We're working closely with our beta customers. We're making progress to improve their overall customer satisfaction and reduce the response time"
    ],
    [
        "Tell me a success story",
        "We're working closely with our beta customers. We're making progress to improve their overall customer satisfaction and reduce the response time"
    ]
]


ask_company_intent = [conversation[0] for conversation in ask_company_data]
ask_customer_intent = [conversation[0] for conversation in ask_customer_data]
ask_doc_intent = [conversation[0] for conversation in ask_doc_data]
ask_price_intent = [conversation[0] for conversation in ask_price_data]
ask_product_intent = [conversation[0] for conversation in ask_product_data]
ask_story_intent = [conversation[0] for conversation in ask_story_data]

general_bot = get_new_bot('general_bot')
general_bot.train(
    "chatterbot.corpus.english",
)

greeting_bot = get_new_bot('greeting_bot')
greeting_bot.train(
    "chatterbot.corpus.english.greetings",
)

identity_bot = get_new_bot('identity_bot')
for conversation in identity_data:
    identity_bot.train(conversation)

company_bot = get_new_bot('company_bot')
all_data = ask_company_data + ask_customer_data + ask_price_data + ask_doc_data + ask_story_data + ask_product_data

for conversation in all_data:
    company_bot.train(conversation)


GREETING_INTENT = 'greetings'
IDENTITY_INTENT = 'identity'
ASK_INTENT_PATTERN = re.compile('ask.*')


def select_bot(in_msg):
    resp = message(wit_token, in_msg)
    print resp['outcomes'][0]['intent']
    if resp['outcomes'][0]['intent'] == GREETING_INTENT:
        return greeting_bot
    elif resp['outcomes'][0]['intent'] == IDENTITY_INTENT:
        return identity_bot
    elif ASK_INTENT_PATTERN.match(resp['outcomes'][0]['intent']):
        return company_bot
    else:
        return general_bot


def chat(question):
    bot = select_bot(question)
    return bot.get_response(question)

if __name__ == '__main__':
    while True:
        in_msg = raw_input()
        out_msg = chat(in_msg)
        print "Buddy bot :> %s" % out_msg
