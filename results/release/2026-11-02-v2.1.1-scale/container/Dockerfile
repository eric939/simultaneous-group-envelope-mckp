FROM python:3.12.12-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv==0.9.5
COPY repo.bundle /tmp/repo.bundle
RUN git clone -q -b experiments/v2.1 /tmp/repo.bundle /work && git -C /work checkout -q 0885e82 && rm /tmp/repo.bundle
WORKDIR /work
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
RUN uv sync --frozen --extra validation --extra dev --python /usr/local/bin/python3.12
ENV PATH=/opt/venv/bin:$PATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
