FROM pytorch/pytorch:latest

RUN apt-get update & apt-get -y install curl

RUN pip install numpy matplotlib transformers huggingface-hub dspy

EXPOSE 11434

RUN curl -fsSL https://ollama.com/install.sh | sh
CMD ["ollama", "serve"]
# RUN ollama pull llama3:8b
