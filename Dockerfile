FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends git build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/counthallu
COPY . .
RUN pip install --no-cache-dir -e .

CMD ["/bin/bash"]
