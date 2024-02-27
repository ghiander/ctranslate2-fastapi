import logging

from fastapi import HTTPException
from functools import wraps
from languagemodels.inference import InvalidTokenException
from languagemodels.inference import InferenceException
from languagemodels.inference import MaxTokensException

logger = logging.getLogger(__name__)


def _format_exception(e):
    return f"{type(e)}: {str(e)}"


def error_handling(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except (InvalidTokenException,
                InferenceException) as e:
            logger.error(e)
            raise HTTPException(status_code=400,
                                detail=_format_exception(e))
        except MaxTokensException as e:
            logger.error(e)
            raise HTTPException(status_code=413,
                                detail=_format_exception(e))
        except Exception as e:
            logger.error(e)
            raise HTTPException(status_code=500,
                                detail=_format_exception(e))
    return wrapper
