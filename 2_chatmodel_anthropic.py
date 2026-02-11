from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
# we dont have an api key of anthropic
load_dotenv()

model = ChatAnthropic(model='claude-3-5-sonnet-20241022')

result = model.invoke('What is the capital of India')

print(result.content)