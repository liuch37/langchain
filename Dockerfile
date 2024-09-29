# Build command: 
# docker build -t rag-streamlit .
# Deploy command:
# docker run -p 8051:8051 rag-streamlit
# app/Dockerfile

FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip3 install -r requirements.txt

EXPOSE 8051

HEALTHCHECK CMD curl --fail http://localhost:8051/_stcore/health

ENTRYPOINT ["streamlit", "run", "ragchatbot.py", "--server.port=8051", "--server.address=0.0.0.0"]