from langchain_community.document_loaders import CSVLoader
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.chroma import Chroma
embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
loader = CSVLoader("FoodBot/Data/menu.csv")
data = loader.load()
db = Chroma(data,embedding_function=embeddings,persist_directory="FoodBot/app/fooddb")