from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()
prompt = PromptTemplate(
    template = "Create 3 medium-level math word problems based on the topic : {topic}",
    input_variables=["topic"]
)
model = ChatOpenAI(model="gpt-5-nano",temperature = 0.8)
parser = StrOutputParser()
#pipeine -> prompt -> model -> parser.
# |->pipe symbol
# runnable operator
chain = prompt | model | parser
result = chain.invoke({"topic":"Time and Work"})
print(result)
# show graph
chain.get_graph().print_ascii()