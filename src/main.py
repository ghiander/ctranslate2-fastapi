import config
import logging
import languagemodels as lm

from fastapi import FastAPI
from model import CompletionQuery, CompletionResponse, ChatQuery
from helpers import prefill_response
from helpers import clean_completion
from helpers import make_message_and_content_str


app = FastAPI(title=config.get_app_title(),
              description=config.get_app_description())
artifact_tup = lm.get_preloaded_artifacts()
model_name = lm.get_model_name()
logger = logging.getLogger(__name__)
logger.info(f"Loaded '{model_name}' model into memory")


@app.get("/health")
async def root():
    return {"message": "Hello World"}


@app.post("/completions", response_model=CompletionResponse)
async def completions(query: CompletionQuery):
    prompt = query.prompt
    completion = lm.do(prompt,
                       preloaded_artifacts=artifact_tup)
    completion = clean_completion(completion)
    response = prefill_response(prompt, completion)
    response["choices"] = [{"text": completion}]
    return response


@app.post("/chat/completions")
async def chat(query: ChatQuery):
    messages = query.messages
    message_str, content_str = \
        make_message_and_content_str(messages)
    completion = lm.chat(message_str,
                         preloaded_artifacts=artifact_tup)
    completion = clean_completion(completion)
    response = prefill_response(content_str, completion)
    response["choices"] = [{
        "message": {
            "role": "assistant",
            "content": completion
        }}]
    return response
