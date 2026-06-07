# from app.services.chat_service import ChatService

# chat = ChatService()

# response, docs = chat.answer(
#     "Who founded the company?",
#     []
# )

# print(response)

from app.services.chat_service import ChatService

chat = ChatService()

history = []

response, docs = chat.answer(
    question="Who founded the company?",
    history=history
)

history.append(
    {
        "role":"user",
        "content":"Who founded the company?"
    }
)

history.append(
    {
        "role":"assistant",
        "content":response
    }
)

response, docs = chat.answer(
    question="When was it founded?",
    history=history
)

print(response)