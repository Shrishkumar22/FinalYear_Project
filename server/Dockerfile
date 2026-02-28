FROM python:3.10

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu

COPY main.py utils.py client1_test.csv teacher_optimized.pth /app/

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]