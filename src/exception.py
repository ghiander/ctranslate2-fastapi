import logging

from fastapi import HTTPException
from functools import wraps
from languagemodels.inference import InvalidTokenException
from languagemodels.inference import InferenceException
from languagemodels.inference import MaxTokensException

logger = logging.getLogger(__name__)


def error_handling(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except (InvalidTokenException,
                InferenceException,
                MaxTokensException) as e:
            logger.error(e)
            raise HTTPException(status_code=400,
                                detail=f"{type(e)}: {str(e)}")
        except Exception as e:
            logger.error(e)
            raise HTTPException(status_code=400,
                                detail=f"{type(e)}: {str(e)}")
    return wrapper
