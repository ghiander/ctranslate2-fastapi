from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello World"}


def test_completions():
    request = {
        "prompt": "What's the first name of the secret agent Bond?"
    }
    response = client.post("/completions", json=request)
    assert response.status_code == 200
    res_json = response.json()
    assert "model" in res_json
    assert "usage" in res_json
    assert "id" in res_json
    assert "choices" in res_json
    assert "james" in res_json["choices"][0]["text"].lower()


def test_chat_simple():
    request = {
        "messages": [
            {
                "role": "system",
                "content": "Respond in one word."
            },
            {
                "role": "user",
                "content": "What's the first name of Bill Gates?"
            }
        ]
    }
    response = client.post("/chat/completions", json=request)
    assert response.status_code == 200
    res_json = response.json()
    assert "model" in res_json
    assert "usage" in res_json
    assert "id" in res_json
    assert "choices" in res_json
    assert "bill" in res_json["choices"][0][
        "message"]["content"].lower()


chain_message = "You have access to the following tools:\n\n" \
    "WebSearch: Searches the web.\n\nUse the following format:\n\n" \
    "Question: the input question you must answer\n" \
    "Action: the action to take, should be one of [nWebSearch]\n" \
    "Action Input: the input to the action\nObservation: " \
    "the result of the action\nThought: I now know the final answer\n" \
    "Final Answer: the final answer to the original input question\n\n" \
    "Begin!\n\nQuestion: What day is today?\nThought:"


def test_chat_chain():
    request = {
        "messages": [
            {
                "role": "system",
                "content": "Answer the following question as best you can."
            },
            {
                "role": "user",
                "content": chain_message
            }
        ]
    }
    response = client.post("/chat/completions", json=request)
    assert response.status_code == 200
    res_json = response.json()
    assert "model" in res_json
    assert "usage" in res_json
    assert "id" in res_json
    assert "choices" in res_json
