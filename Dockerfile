FROM python:3.9

RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx libglib2.0-0

RUN pip install --upgrade pip

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

COPY . /app

WORKDIR /app


CMD [ "python3" , "-m" , "flask" , "run" , "--host=0.0.0.0" ]