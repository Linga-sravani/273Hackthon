from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

import logic

app = FastAPI()

origins = ["*",]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Welcome to the RAG Orchestration API"}


class MessageRequest(BaseModel):
    message: str


@app.post("/get-answer")
def get_answer(request: MessageRequest):
    received = logic.get_answer(request.message)
    return {"message": received["data"]["output_text"]}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8099)
