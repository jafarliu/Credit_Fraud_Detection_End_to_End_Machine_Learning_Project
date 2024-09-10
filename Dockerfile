FROM python:3.9.18
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE $PORT
CMD python app.py --bind 0.0.0.0:$PORT 