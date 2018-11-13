FROM ubuntu:18.10

LABEL maintainer="Robin Rojowiec <nijou49@gmail.com>"

RUN apt-get update -y && \
    apt-get install -y python3 python3-pip python3-dev

# We copy just the requirements.txt first to leverage Docker cache
COPY ./scripts/requirements.txt /app/requirements.txt

WORKDIR /app

RUN pip3 install -r requirements.txt

COPY . /app

ENTRYPOINT [ "python3" ]

CMD [ "flask_server.py" ]
