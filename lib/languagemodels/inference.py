import re
import os
import logging

from typing import List
from languagemodels.models import get_artifacts, get_model_info
from languagemodels.bootstrap import get_artifact_dir


# Configure logging logic
log = logging.getLogger(__name__)
if os.environ.get("LOGGING_LEVEL") == "DEBUG":
    logging.basicConfig(level=logging.DEBUG)
else:
    logging.basicConfig(level=logging.INFO)


class InferenceException(Exception):
    pass


def load_artifacts_into_memory(model_info):
    """Returns the preloaded tokenizer and model as a tuple."""
    artifact_dir = get_artifact_dir()
    return get_artifacts(artifact_dir, model_info)


def generate(
    instructions: List[str],
    max_tokens: int = 200,
    temperature: float = 0.1,
    topk: int = 1,
    repetition_penalty: float = 1.3,
    prefix: str = "",
    suppress: List[str] = [],
    preloaded_artifacts: tuple = None
):
    """Generates completions for a prompt

    This may use a local model, or it may make an API call to an external
    model if API keys are available.

    >>> generate(["What is the capital of France?"])
    ... # doctest: +ELLIPSIS
    ['...Paris...']
    """
    artifact_dir = get_artifact_dir()
    model_info = get_model_info()
    if not preloaded_artifacts:
        log.debug("Artifacts are lazy-loaded into memory")
        tokenizer, model = get_artifacts(artifact_dir,
                                         model_info)
    else:
        tokenizer = preloaded_artifacts[0]
        model = preloaded_artifacts[1]

    suppress = [tokenizer.encode(s, add_special_tokens=False).tokens
                for s in suppress]
    fmt = model_info.get("prompt_fmt", "{instruction}")
    prompts = [fmt.replace("{instruction}", inst)
               for inst in instructions]

    outputs_ids = []
    prefix = tokenizer.encode(prefix, add_special_tokens=False).tokens
    results = model.translate_batch(
        [tokenizer.encode(p).tokens for p in prompts],
        target_prefix=[prefix] * len(prompts),
        repetition_penalty=repetition_penalty,
        max_decoding_length=max_tokens,
        sampling_temperature=temperature,
        sampling_topk=topk,
        suppress_sequences=suppress,
        beam_size=1,
    )
    outputs_tokens = [r.hypotheses[0] for r in results]
    for output in outputs_tokens:
        outputs_ids.append([tokenizer.token_to_id(t) for t in output])

    return [tokenizer.decode(i, skip_special_tokens=True).lstrip()
            for i in outputs_ids]


def list_tokens(prompt):
    """Generates a list of tokens for a supplied prompt

    >>> list_tokens("Hello, world!") # doctest: +SKIP
    [('▁Hello', 8774), (',', 6), ('▁world', 296), ('!', 55)]

    >>> list_tokens("Hello, world!")
    ... # doctest: +ELLIPSIS
    [('...Hello', ...), ... ('...world', ...), ...]
    """
    artifact_dir = get_artifact_dir()
    model_info = get_model_info()
    tokenizer, model = get_artifacts(artifact_dir, model_info)
    output = tokenizer.encode(prompt, add_special_tokens=False)
    tokens = output.tokens
    ids = output.ids
    return list(zip(tokens, ids))


def rank_instruct(inputs, targets):
    """Sorts a list of targets by their probabilities

    >>> rank_instruct(["Classify positive or negative: \
        I love python. Classification:"],
    ... ['positive', 'negative'])
    [['positive', 'negative']]

    >>> rank_instruct(["Classify fantasy or documentary: "
    ... "The wizard raised their want. Classification:"],
    ... ['fantasy', 'documentary'])
    [['fantasy', 'documentary']]

    >>> rank_instruct(["Say six", "Say seven"], ["six", "seven"])
    [['six', 'seven'], ['seven', 'six']]
    """
    artifact_dir = get_artifact_dir()
    model_info = get_model_info()
    tokenizer, model = get_artifacts(artifact_dir, model_info)
    targ_tok = [tokenizer.encode(t, add_special_tokens=False).tokens
                for t in targets]
    targ_tok *= len(inputs)

    in_tok = []
    for input in inputs:
        toks = [tokenizer.encode(input, add_special_tokens=False).tokens]
        in_tok += toks * len(targets)

    if "Generator" in str(type(model)):
        scores = model.score_batch([i+t for i, t in zip(in_tok, targ_tok)])
    else:
        scores = model.score_batch(in_tok, target=targ_tok)

    ret = []
    for i in range(0, len(inputs) * len(targets), len(targets)):
        logprobs = [sum(r.log_probs) for r in scores[i:i+len(targets)]]
        results = sorted(zip(targets, logprobs), key=lambda r: -r[1])
        ret.append([r[0] for r in results])

    return ret


def parse_chat(prompt):
    """Converts a chat prompt using special tokens to a plain-text prompt

    This is useful for prompting generic models that have not been fine-tuned
    for chat using specialized tokens.

    >>> parse_chat('User: What time is it?')
    ... # doctest: +ELLIPSIS
    Traceback (most recent call last):
        ...
    languagemodels.inference.InferenceException:\
 Chat prompt must end with 'Assistant:'

    >>> parse_chat('''User: What time is it?
    ...
    ...               Assistant:''')
    [{'role': 'user', 'content': 'What time is it?'}]

    >>> parse_chat('''
    ...              A helpful assistant
    ...
    ...              User: What time is it?
    ...
    ...              Assistant:
    ...              ''')
    [{'role': 'system', 'content': 'A helpful assistant'},\
 {'role': 'user', 'content': 'What time is it?'}]

    >>> parse_chat('''
    ...              A helpful assistant
    ...
    ...              User: What time is it?
    ...
    ...              Assistant: The time is
    ...              ''')
    Traceback (most recent call last):
        ....
    languagemodels.inference.InferenceException:\
 Final assistant message must be blank

    >>> parse_chat('''
    ...              A helpful assistant
    ...
    ...              User: First para
    ...
    ...              Second para
    ...
    ...              Assistant:
    ...              ''')
    [{'role': 'system', 'content': 'A helpful assistant'},\
 {'role': 'user', 'content': 'First para\\n\\nSecond para'}]

    >>> parse_chat('''
    ...              A helpful assistant
    ...
    ...              User: What time is it?
    ...
    ...              InvalidRole: Nothing
    ...
    ...              Assistant:
    ...              ''')
    Traceback (most recent call last):
        ....
    languagemodels.inference.InferenceException: Invalid chat role: invalidrole
    """

    if not re.match(r"^\s*\w+:", prompt):
        prompt = "System: " + prompt

    prompt = "\n\n" + prompt

    chunks = re.split(r"[\r\n]\s*(\w+):", prompt, flags=re.M)
    chunks = [m.strip() for m in chunks if m.strip()]

    messages = []

    for i in range(0, len(chunks), 2):
        role = chunks[i].lower()

        try:
            content = chunks[i + 1]
            content = re.sub(r"\s*\n\n\s*", "\n\n", content)
        except IndexError:
            content = ""
        messages.append({"role": role, "content": content})

    for message in messages:
        if message["role"] not in ["system", "user", "assistant"]:
            raise InferenceException(f"Invalid chat role: {message['role']}")

    if messages[-1]["role"] != "assistant":
        raise InferenceException("Chat prompt must end with 'Assistant:'")

    if messages[-1]["content"] != "":
        raise InferenceException("Final assistant message must be blank")

    return messages[:-1]
