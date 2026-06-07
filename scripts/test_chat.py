from app.services.chat_service import ChatService

chat = ChatService()

response, docs = chat.answer(
    "Who founded the company?",
    []
)

print(response)