from pydantic import BaseModel
from pydantic import conlist
from typing import List
from enum import Enum


class CompletionQuery(BaseModel):
    prompt: str


class Role(str, Enum):
    USER = 'user'
    SYSTEM = 'system'
    ASSISTANT = "assistant"


class RoleContentChat(BaseModel):
    role: Role
    content: str


class ChatQuery(BaseModel):
    # Defines a min and max length for 'messages'
    messages: conlist(
        RoleContentChat, min_length=1, max_length=5)


class UsageResponse(BaseModel):
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int


class TextCompletion(BaseModel):
    text: str


class BaseResponse(BaseModel):
    id: str
    model: str
    usage: UsageResponse


class CompletionResponse(BaseResponse):
    choices: List[TextCompletion]
