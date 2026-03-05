FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu --timeout 300 \
    && pip install -r requirements.txt

COPY main.py ssf_student.py utils.py client1_test.csv /app/

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9000", "--loop", "asyncio"]