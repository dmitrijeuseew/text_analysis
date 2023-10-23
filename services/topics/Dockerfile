FROM deeppavlov/base-gpu:0.17.6

WORKDIR /app

RUN apt-get update && apt-get install git -y

COPY /services/topics/requirements.txt /src/requirements.txt

RUN pip install -r /src/requirements.txt

RUN pip install deeppavlov==1.1.1

RUN pip uninstall -y websockets

RUN pip install websockets

WORKDIR /src

COPY /services/ner /src

CMD python server.py
