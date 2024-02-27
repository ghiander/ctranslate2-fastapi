import config
import logging
import languagemodels as lm

from fastapi import FastAPI
from exception import error_handling
from model import CompletionQuery
from model import CompletionResponse
from model import ChatQuery
from helpers import prefill_response
from helpers import clean_completion
from helpers import make_message_and_content_str
from helpers import serialize_messages


app = FastAPI(title=config.get_app_title(),
              description=config.get_app_description())
artifact_tup = lm.get_preloaded_artifacts()
model_name = lm.get_model_name()
logger = config.get_logger(__name__)
logger.info(f"Loaded '{model_name}' model into memory")

@app.get("/health")
async def root():
    return {"message": "Hello World"}


@app.post("/completions", response_model=CompletionResponse)
@error_handling
async def completions(query: CompletionQuery):
    logger.debug(query)
    prompt = query.prompt
    completion = lm.do(prompt,
                       preloaded_artifacts=artifact_tup)
    completion = clean_completion(completion)
    response = prefill_response(prompt, completion)
    response["choices"] = [{"text": completion}]
    return response


@app.post("/chat/completions")
@error_handling
async def chat(query: ChatQuery):
    logger.debug(query)
    messages = query.messages
    _, content_str = \
        make_message_and_content_str(messages)
    messages_dict = serialize_messages(messages)
    completion = lm.chat_from_dict(messages_dict,
                                   preloaded_artifacts=artifact_tup)
    completion = clean_completion(completion)
    response = prefill_response(content_str, completion)
    response["choices"] = [{
        "message": {
            "role": "assistant",
            "content": completion
        }}]
    return response
