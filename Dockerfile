FROM python:3.9-bullseye

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir --progress-bar off -r requirements.txt

EXPOSE 5000

CMD ["python", "app.py"]