from langchain.chat_models import ChatOpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain_openai import OpenAI, ChatOpenAI
from dotenv import load_dotenv

load_dotenv()
chat= ChatOpenAI()

query= """Write 10 beverages for north indian food outlett.  Below is the format for the output:

Drink name: <Dish name>
Price: <Price in Rupees>
description: <Description>
Ingredients: <Ingredients>
is_vegetarian: <yes or no>
dish course: <appetizer, main course, and dessert>"""


query = """I am working with a python project where i have the following directory
home/utils/ll_utils.py
home/utils2/ll_utils2.py
home/main.py

and ll_utils.py is importing a calls from ll_utils2.py uisng this `from utils2.ll_utils import Class1` its thorwing error
ModuleNotFoundError. Why this is happening ?

I am running from home this command:
python3 utils/ll_utils.py
"""

resp = ""
for chunk in chat.stream(query):
    print(chunk.content, end="", flush=True)
    resp += chunk.content