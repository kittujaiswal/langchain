from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()
#step 1- Generate a detial travel itineray
prompt1 = PromptTemplate(
    template = "Create a deatiled 2-day travel itineray for {place}. Include timings,food recommendations,and must-visit attractions.",
Input_variables = ['place']
)
# step 2 - convert that itineary into a short 5-point guide
prompt2 = PromptTemplate(
    template="Convert the following itineary into aa crisp 5-point travel guide: \n{text}",
    input_variables=['text']
)
model = ChatOpenAI(model="gpt-5-nano",temperature = 0.8)
parser = StrOutputParser()
chain = prompt1 | model | parser | prompt2 | model | parser
result = chain.invoke({"place":"Jaipur"})
print(result)
# show graph
chain.get_graph().print_ascii()