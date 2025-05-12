FROM python:3.10-slim

# Install git and other necessary tools
RUN apt-get update && apt-get install -y git && apt-get clean

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "2000"]