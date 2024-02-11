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


def test_chat():
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
