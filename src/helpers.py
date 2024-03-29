import random
import string
import languagemodels as lm


PRIMITIVES = (bool, str, int, float, type(None))


def is_primitive_strict(obj):
    """Strict primitive type check."""
    return type(obj) in PRIMITIVES


def serialize_messages(messages):
    """Serialises a FastAPI dictionary
    by converting model objects into
    their corresponding values."""
    messages_serial = list()
    for m in messages:
        message = dict()
        for k, v in m.__dict__.items():
            # If not serialisable, get the value attribute
            if not is_primitive_strict(v):
                message[k] = v._value_
            else:
                message[k] = v
        messages_serial.append(message)
    return messages_serial


def _generate_random_id(N=10):
    """Generates a random alphanumeric
    string of N characters."""
    return ''.join(
        random.choices(
            string.ascii_uppercase + string.digits, k=N))


def _calculate_tokens(prompt, completion):
    return lm.count_tokens(prompt), lm.count_tokens(completion)


def prefill_response(prompt, completion):
    """Boilerplate for generating a response
    with the same schema of that from OpenAI."""
    prompt_tks, completion_tks = \
        _calculate_tokens(prompt, completion)
    return {
        "id": _generate_random_id(),
        "model": lm.get_model_name(),
        "usage": {
            "completion_tokens": completion_tks,
            "prompt_tokens": prompt_tks,
            "total_tokens": completion_tks + prompt_tks
        }
    }


def _remove_surrounding_quotes(s):
    if s.startswith("\""):
        s = s[1:]
    if s.endswith("\""):
        s = s[:-1]
    return s


def clean_completion(completion):
    completion = _remove_surrounding_quotes(completion)
    return completion


def make_message_and_content_str(messages):
    """Converts a list of message objects into
    a string prompt and a concatenated string."""
    message_str = ""
    content_str = ""
    for m in messages:
        message_str += f"{m.role.capitalize()}: " \
            f"{m.content.capitalize()}\n\n"
        content_str += f"{m.content} "
    message_str += "Assistant: "
    content_str = content_str[:-1]
    return message_str, content_str
