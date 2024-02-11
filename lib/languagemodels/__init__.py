import datetime
from typing import overload

from languagemodels.config import config
from languagemodels.models import get_model_info
from languagemodels.inference import (
    generate,
    rank_instruct,
    parse_chat,
    list_tokens,
    load_artifacts_into_memory
)


def get_model_name() -> str:
    return config["name"]


def complete(prompt: str) -> str:
    """Provide one completion for a given open-ended prompt

    :param prompt: Prompt to use as input to the model
    :return: Completion returned from the language model

    Examples:

    >>> complete("Luke thought that he") #doctest: +SKIP
    'was going to be a doctor.'

    >>> complete("There are many mythical creatures who") #doctest: +SKIP
    'are able to fly'

    >>> complete("She hid in her room until") #doctest: +SKIP
    'she was sure she was safe'
    """

    result = generate(
        ["Write a sentence"], prefix=prompt,
        max_tokens=config["max_tokens"], temperature=0.7, topk=40
    )[0]

    if result.startswith(prompt):
        prefix_length = len(prompt)
        return result[prefix_length:]
    else:
        return result


def get_preloaded_artifacts():
    """Returns tokenizer and model as
    objects already loaded in memory
    for fast inference"""
    model_info = get_model_info()
    return load_artifacts_into_memory(model_info)


def _refine_response_punctuation(results):
    # TODO: sic - may be removed?
    for i, result in enumerate(results):
        if len(result.split()) == 1:
            results[i] = result.title()

            if result[-1] not in (".", "!", "?"):
                results[i] = results[i] + "."
    return results


@overload
def do(prompt: list) -> list:
    ...


@overload
def do(prompt: str) -> str:
    ...


def do(prompt, choices=None, preloaded_artifacts=None):
    """Follow a single-turn instructional prompt

    :param prompt: Instructional prompt(s) to follow
    :param choices: If provided, outputs are restricted to values in choices
    :return: Completion returned from the language model

    Note that this function is overloaded to return a list of results if
    a list if of prompts is provided and a single string if a single
    prompt is provided as a string

    Examples:

    >>> do("Translate Spanish to English: Hola mundo!") #doctest: +SKIP
    'Hello world!'

    >>> do("Pick the sport from the list: baseball, texas, chemistry")
    'Baseball.'

    >>> do("Is the following positive or negative: I love Star Trek.")
    'Positive.'

    >>> do(["Pick the sport from the list: baseball, texas, chemistry"] * 2)
    ['Baseball.', 'Baseball.']

    >>> do(["Say red", "Say blue"], choices=["red", "blue"])
    ['red', 'blue']
    """

    prompts = [prompt] if isinstance(prompt, str) else prompt

    if not preloaded_artifacts:
        if choices:
            results = [r[0] for r in rank_instruct(prompts, choices)]
        else:
            results = generate(prompts,
                               max_tokens=config["max_tokens"],
                               topk=1)

    else:
        results = generate(prompts,
                           max_tokens=config["max_tokens"],
                           topk=1,
                           preloaded_artifacts=preloaded_artifacts)

    if not choices:
        results = _refine_response_punctuation(results)

    return results[0] if isinstance(prompt, str) else results


def chat(prompt: str, preloaded_artifacts=None) -> str:
    """Get new message from chat-optimized language model

    The `prompt` for this model is provided as a series of messages as a single
    plain-text string. Several special tokens are used to delineate chat
    messages.

    - `system:` - Indicates the start of a system message providing
    instructions about how the assistant should behave.
    - `user:` - Indicates the start of a prompter (typically user)
    message.
    - `assistant:` - Indicates the start of an assistant message.

    A complete prompt may look something like this:

    ```
    Assistant is helpful and harmless

    User: What is the capital of Germany?

    Assistant: The capital of Germany is Berlin.

    User: How many people live there?

    Assistant:
    ```

    The completion from the language model is returned.

    :param message: Prompt using formatting described above
    :return: Completion returned from the language model

    Examples:

    >>> chat('''
    ...      System: Respond as a helpful assistant. It is 5:00pm.
    ...
    ...      User: What time is it?
    ...
    ...      Assistant:
    ...      ''')
    ... # doctest: +ELLIPSIS
    '...5:00pm...'
    """

    messages = parse_chat(prompt)

    # Suppress starts of all assistant messages to avoid repeat generation
    suppress = [
        "Assistant: " + m["content"].split(" ")[0]
        for m in messages
        if m["role"] in ["assistant", "user"]
    ]

    # Suppress all user messages to avoid repeating them
    suppress += [m["content"] for m in messages if m["role"] == "user"]

    system_msgs = [m for m in messages if m["role"] == "system"]
    assistant_msgs = [m for m in messages if m["role"] == "assistant"]
    user_msgs = [m for m in messages if m["role"] == "user"]

    # The current model is tuned on instructions and tends to get
    # lost if it sees too many questions
    # Use only the most recent user and assistant message for context
    # Keep all system messages
    messages = system_msgs + assistant_msgs[-1:] + user_msgs[-1:]

    rolemap = {
        "system": "System",
        "user": "Question",
        "assistant": "Assistant",
    }

    messages = [f"{rolemap[m['role']]}: {m['content']}" for m in messages]

    prompt = "\n\n".join(messages) + "\n\n" + "Assistant:"

    if prompt.startswith("System:"):
        prompt = prompt[7:].strip()

    response = generate(
        [prompt],
        max_tokens=config["max_tokens"],
        repetition_penalty=1.3,
        temperature=0.3,
        topk=40,
        prefix="Assistant:",
        suppress=suppress,
        preloaded_artifacts=preloaded_artifacts
    )[0]

    # Remove duplicate assistant being generated
    if response.startswith("Assistant:"):
        response = response[10:]

    return response.strip()


def extract_answer(question: str, context: str) -> str:
    """Extract an answer to a `question` from a provided `context`

    The returned answer will always be a substring extracted from `context`.
    It may not always be a correct or meaningful answer, but it will never be
    an arbitrary hallucination.

    :param question: A question to answer using knowledge from context
    :param context: Knowledge used to answer the question
    :return: Answer to the question.

    Examples:

    >>> context = "There is a green ball and a red box"
    >>> extract_answer("What color is the ball?", context).lower()
    ... # doctest: +ELLIPSIS
    '...green...'
    """

    return generate([f"{context}\n\n{question}"])[0]


def classify(doc: str, label1: str, label2: str) -> str:
    """Performs binary classification on an input

    :param doc: A plain text input document to classify
    :param label1: The first label to classify against
    :param label2: The second label to classify against
    :return: The closest matching class. The return value will always be
    `label1` or `label2`

    Examples:

    >>> classify("I love you!","positive","negative")
    'positive'
    >>> classify("That book was good.","positive","negative")
    'positive'
    >>> classify("That movie was scary.","positive","negative")
    'negative'
    >>> classify("The submarine is diving", "ocean", "land")
    'ocean'
    """

    results = rank_instruct(
        [f"Classify as {label1} or {label2}: {doc}\n\nClassification:"],
        [label1, label2]
    )

    return results[0][0]


def get_date() -> str:
    """Returns the current date and time in natural language

    >>> get_date() # doctest: +SKIP
    'Friday, May 12, 2023 at 09:27AM'
    """
    now = datetime.datetime.now()
    return now.strftime("%A, %B %d, %Y at %I:%M%p")


def print_tokens(prompt: str) -> None:
    """Prints a list of tokens in a prompt

    :param prompt: Prompt to use as input to tokenizer
    :return: Nothing

    Examples:

    >>> print_tokens("Hello world")
    ' Hello' (token 8774)
    ' world' (token 296)

    >>> print_tokens("Hola mundo")
    ' Hol' (token 5838)
    'a' (token 9)
    ' mun' (token 13844)
    'd' (token 26)
    'o' (token 32)
    """

    tokens = list_tokens(prompt)

    for token in tokens:
        print(f"'{token[0].replace('▁',' ')}' (token {token[1]})")


def count_tokens(prompt: str) -> None:
    """Counts tokens in a prompt

    :param prompt: Prompt to use as input to tokenizer
    :return: Nothing

    Examples:

    >>> count_tokens("Hello world")
    2

    >>> count_tokens("Hola mundo")
    5
    """

    return len(list_tokens(prompt))


def set_max_ram(value):
    """Sets max allowed RAM

    This value takes priority over environment variables

    Returns the numeric value set in GB

    >>> set_max_ram(16)
    16.0

    >>> set_max_ram('512mb')
    0.5
    """

    config["max_ram"] = value

    return config["max_ram"]


if __name__ == "__main__":
    import doctest
    doctest.testmod()
