FROM python:3.10-slim

WORKDIR /workspace

COPY requirements.txt .
RUN \
pip install --upgrade pip \
&& pip install -r requirements.txt

# Opcional: instala herramientas útiles
RUN apt-get update && apt-get install -y tree

# Default command (útil para pruebas locales)
CMD ["bash"]
