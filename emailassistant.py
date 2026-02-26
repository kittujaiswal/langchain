# Logic:
# If the user text is about professional, generate:
# important using Model-1
# action using Model-2
# If the user text is about spam, generate:
# spam reason using Model-1
# block recommendation using Model-2
# Then merge both outputs into a single documen

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnableParallel
load_dotenv()
model1 = ChatOpenAI(model="gpt-5-nano", temperature=0.8)
model2 = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",   
)
parser = StrOutputParser()
# ---- PROMPTS FOR DIFFERENT DOMAINS ----
professional_reply = PromptTemplate(
    template="Write a professional email from the text:\n{text}",
    input_variables=["text"]
)

action_items = PromptTemplate(
    template="Write all the important action to take from the text:\n{text}",
    input_variables=["text"]
)

health_notes = PromptTemplate(
    template="Write simple and short HEALTH notes from the text:\n{text}",
    input_variables=["text"]
)

spam_reason = PromptTemplate(
    template="Write the reason why email is spam from the:\n{text}",
    input_variables=["text"]
)
block_recommendation = PromptTemplate(
    template="Write the suggestion to block email:\n{text}",
    input_variables=["text"]
)
prompt = PromptTemplate(
    template="Give the best suggestion to the user"
)


# ---- CONDITIONAL LOGIC (BRANCH) ----
def classify_topic(input_dict):
    text = input_dict["text"].lower()
    if any(word in text for word in ["professional", "action"]):
        return "important"
    return "spam"


important_mail = RunnableParallel({
    "professional_reply" : professional_reply | model1 | parser,
    "action_items" : action_items | model2 | parser
})

spam_mail = RunnableParallel({
    "spam_reason": spam_reason | model1 | parser,
    "block_recommendation": block_recommendation | model2 | parser
})

conditional_chain = RunnableBranch(
    (lambda x: classify_topic(x) == "important",important_mail),
    spam_mail   # default condition
)

# ---- FINAL MERGE ----
final_chain = conditional_chain | (prompt | model2 | parser)


# ---- TESTING INPUT ----
text = """
I hope this message finds important for you. I am reaching out to provide an update
on the current progress and to ensure we remain aligned on the next steps.
"""

result = final_chain.invoke({"text": text})
print(result)


final_chain.get_graph().print_ascii()