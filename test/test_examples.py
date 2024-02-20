import time
import languagemodels as lm


# TEST COMPLETION
query = "Tell me two songs by Radiohead"
print(query)
start = time.time()
print(lm.do(query))
end = time.time()
print(end - start)
print(lm.get_model_info())

# TEST CHAT
chat_query = """
System: Respond red.

User: What color is the sky?

Assistant: """
print(chat_query)
start = time.time()
print(lm.chat(chat_query))
end = time.time()
print(end - start)

# TEST CHAT FROM DICT
chat_dict_query = [
    {
        "role": "system",
        "content": "Respond 'red'."
    },
    {
        "role": "user",
        "content": "What color is the sky?`"
    }
]
print(lm.chat_from_dict(chat_dict_query))

# PRELOAD ARTIFACTS
start = time.time()
artifact_tup = lm.get_preloaded_artifacts()
end = time.time()
print(end - start)

# TEST PRELOADED COMPLETION
start = time.time()
print(lm.do(query,
      preloaded_artifacts=artifact_tup))
end = time.time()
print(end - start)

# TEST PRELOADED CHAT
start = time.time()
print(lm.chat(chat_query,
      preloaded_artifacts=artifact_tup))
end = time.time()
print(end - start)
