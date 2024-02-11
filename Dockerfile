FROM python:3.10.13-slim-bullseye

# Install dependencies
COPY env/ /env
RUN python -m pip install -r /env/requirements.txt \
    && pip cache purge

# Configure libraries
COPY lib/ /lib
COPY src/ /src
ENV PYTHONPATH "${PYTHONPATH}:/lib"
ENV PYTHONPATH "${PYTHONPATH}:/src"

# Configure wrapper
ENV LLM_ARTIFACT_DIR="/artifacts"
CMD uvicorn --app-dir src main:app --host 0.0.0.0