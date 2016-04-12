__author__ = 'Deyang'

import os
from wit import message

token = 'QUCDCX7MQX4FLYGONBEYLGDHKSTIUFTQ'

resp = message(token, 'Hi')
print resp['outcomes']

