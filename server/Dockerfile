FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu --timeout 300 \
    && pip install -r requirements.txt

COPY main.py utils.py trust.py teacher_optimized.pth /app/

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--loop", "asyncio"]