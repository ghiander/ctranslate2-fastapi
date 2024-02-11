import languagemodels as lm

from tokenizers import Tokenizer
from ctranslate2._ext import Translator
from helpers import make_message_and_content_str
from model import Role
from model import RoleContentChat
from utils import time_function


completion_query = "What is the name of the " \
    "biggest planet in the solar system?"

chat_query = """
System: Respond 'red'.

User: What color is the sky?

Assistant: """

artifact_tup = lm.get_preloaded_artifacts()


def test_load_artifacts():
    tokenizer = artifact_tup[0]
    assert isinstance(tokenizer, Tokenizer)
    model = artifact_tup[1]
    assert isinstance(model, Translator)
    return artifact_tup


def test_completions_lazy_loading():
    res = lm.do(completion_query)
    assert "jupiter" in res.lower()


def test_completions_in_memory():
    res = lm.do(completion_query,
                preloaded_artifacts=artifact_tup)
    assert "jupiter" in res.lower()


def test_relative_speed_completions():
    """In-memory inference must be
    faster than lazy-loaded inference."""
    assert time_function(
        test_completions_in_memory) < \
        time_function(test_completions_lazy_loading)


def test_chat_lazy_loading():
    res = lm.chat(chat_query)
    assert "red" in res.lower()


def test_chat_in_memory():
    res = lm.chat(chat_query,
                  preloaded_artifacts=artifact_tup)
    assert "red" in res.lower()


def test_relative_speed_chat():
    """In-memory inference must be
    faster than lazy-loaded inference."""
    assert time_function(
        test_chat_in_memory) < \
        time_function(test_chat_lazy_loading)


def test_count_tokens():
    assert lm.count_tokens(completion_query) == 13


def test_get_model_name():
    assert isinstance(lm.get_model_name(), str)


def test_chat_input_conversion():
    """"Conversion of structured input
    into concatenated string."""
    messages = [RoleContentChat(
        role=Role.USER,
        content="Foo"), RoleContentChat(
            role=Role.SYSTEM,
            content="Bar")]
    message_str, content_str = \
        make_message_and_content_str(messages)
    expected_message_str = "User: Foo\n\nSystem: Bar\n\nAssistant: "
    assert message_str == expected_message_str
    assert content_str == "Foo Bar"
