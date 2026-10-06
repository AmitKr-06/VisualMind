import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)
client = OpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",
)

models = client.models.list()
for m in sorted(models.data, key=lambda x: x.id):
    print(f"  {m.id}")