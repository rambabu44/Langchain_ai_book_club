from langchain_community.chat_models.ollama import ChatOllama

llm = ChatOllama(model='phi3')
from langchain.schema.runnable import RunnableMap
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.chroma import Chroma

embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

def intent(query:str):
    templateIntent = """
    Classify the intent name by understaning the intent description for the query inside triple backticks.

    <chitchat> : <If the query's intent is simple chitchat like greetings, appreciation and feedback.>
    <foodInquiry>: <If the query's intent is regarding inquiry for food items.>

    query inside triple backticks :```{query}```
    Rspond with intent name only without < > 
    """
    promptIntent = ChatPromptTemplate.from_template(template=templateIntent)

    chainIntentClassification = RunnableMap({
        "query": lambda x: x["query"]
    }) | promptIntent | llm | StrOutputParser()
    return chainIntentClassification.invoke({'query':query})


def chitchat(query:str):
    templateChitchat = """
    You are a restaurant bot working as a waiter. You are provided with simple chitchat query. 
    Respond to the customer's query according to your role. Improvise if necessary.

    query inside triple backticks :```{query}```

    Respond in a humanly manner.
    """
    promptChitchat = ChatPromptTemplate.from_template(template=templateChitchat)

    chainChitchat = RunnableMap({
        "query": lambda x: x["query"]
    }) | promptChitchat | llm | StrOutputParser()
    return chainChitchat.invoke({'query':query})


def foodInquiry(query:str):
    templatefoodInquiry = """
    You are a waiter at the restuarant.
    You are provided with relevant food information along with the customer's query.
    Address to the query according to your role from the information provided to you.
    query inside triple backticks :```{query}```
    context inside double backticks :``{context}``
    Respond in a humanly manner.
    """
    promptfoodInquiry = ChatPromptTemplate.from_template(template=templatefoodInquiry)
    db = Chroma(embedding_function=embeddings,persist_directory="FoodBot/app/fooddb")
    chainfoodInquiry = RunnableMap({
        "query": lambda x: x["query"],
        "context": lambda x: db.similarity_search(x['query'])
    }) | promptfoodInquiry | llm | StrOutputParser()
    return chainfoodInquiry.invoke({'query':query})





