__author__ = 'Deyang'
from chatterbot import ChatBot


chatterbot = ChatBot(
    "Buddy",
    logic_adapters=[
        "chatterbot.adapters.logic.ClosestMatchAdapter",
        "chatterbot.adapters.logic.ClosestMeaningAdapter"
    ],
    io_adapters=[
        "chatterbot.adapters.io.NoOutputAdapter"
    ]
)

chatterbot.train(
    "chatterbot.corpus.english.greetings",
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
    ]
]

identity_intent = [conversation[0] for conversation in identity_data]

print identity_intent

company_data = [
    [
        "What is Buddy AI?",
        "Buddy AI is a startup building AI technologies for modernized customer communication."
    ],
    [
        "Tell me more about Buddy",
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
        "Who are your customers?",
        "We are opening a beta program at present and we work very closely with these companies.  I can't release them over a chat, but if you could give me some more details I'd be happy to get some further information over to you."
    ],
    [
        "Who is using Buddy now?",
        "We are opening a beta program at present and we work very closely with these companies.  I can't release them over a chat, but if you could give me some more details I'd be happy to get some further information over to you."
    ],
    [
        "Can you show me some documentations?",
        "Sure. You can more at everbuddy.io"
    ],
    [
        "Do you have an API?",
        "We have a full REST API. If you are interested in learning more, just respond and I'll get one of the team"
    ],
    [
        "How does it charge?",
        "Buddy AI is currently in private beta and pricing is dependent on multiple variables. To learn more about our beta program, drop us an e-mail to info@everbuddy.io"
    ],
    [
        "What is the price model?",
        "Buddy AI is currently in private beta and pricing is dependent on multiple variables. To learn more about our beta program, drop us an e-mail to info@everbuddy.io"
    ],
    [
        "What are the technologies used here?",
        "I'm using Natural Language Processing to understand your questions and machine learning to find the best answer"
    ],
    [
        "What are the features?",
        "Buddy AI takes any ubiquitous data (forum, customer emails, dev docs, FAQs, internal knowledge base, etc), and then I will learn from the data and provide fully-automated or suggested answers to customer questions."
    ],
]

ask_company_intent = [conversation[0] for conversation in company_data[0:5]]
ask_customer_intent = [conversation[0] for conversation in company_data[5:7]]
ask_doc_intent = [conversation[0] for conversation in company_data[7:9]]
ask_price_intent = [conversation[0] for conversation in company_data[9:11]]
ask_product_intent = [conversation[0] for conversation in company_data[11:13]]


if __name__ == '__main__':
    while True:
        in_msg = raw_input()
        out_msg = chatterbot.get_response(in_msg)
        print out_msg
