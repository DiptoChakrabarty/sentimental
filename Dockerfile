FROM  python:alpine
RUN mkdir app
WORKDIR  /app
COPY  .  .
RUN  pip install -r requirements.txt
CMD ["python","app.py"]