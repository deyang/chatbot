__author__ = 'Deyang'


from wit import message

token = 'QUCDCX7MQX4FLYGONBEYLGDHKSTIUFTQ'

resp = message(token, 'Hi')
print resp['outcomes']