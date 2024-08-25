from fastapi import FastAPI
from FoodBot.app.intentclassification import intent,chitchat,foodInquiry
app = FastAPI()

@app.get("/bot")
def parsequery(query:str):
    intentName = intent(query).lower().lstrip()
    print(intentName)
    if intentName in ['chitchat','<chitchat>']:
        result = chitchat(query)
    elif intentName in ['foodinquiry','<foodinquiry>']:
        result = foodInquiry(query)
    else:
        return "Intent not classified properly"
    return result 