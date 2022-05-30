# STAGE 1 #
FROM python:3.9.12-slim-buster as build

RUN apt-get update && \
    apt-get install \
    --no-install-recommends -y \
    git \
    unixodbc \
    unixodbc-dev \
    freetds-dev \
    python3-dev \
    freetds-bin \
    tdsodbc \
    gcc \
    g++ \
    --reinstall build-essential \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /code/requirements.txt
RUN pip3 install --no-cache --upgrade pip && \
    pip3 install --no-cache -r /code/requirements.txt

RUN git clone https://github.com/asyncee/python-obscene-words-filter
RUN  python /python-obscene-words-filter/setup.py install

# STAGE 2#
FROM python:3.9.12-slim-buster

RUN apt-get update && \
    apt-get install \
    --no-install-recommends -y \
    unixodbc \
    unixodbc-dev \
    freetds-dev \
    freetds-bin \
    tdsodbc \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

RUN echo "[FreeTDS]\n\
Description = FreeTDS unixODBC Driver\n\
Driver = /usr/lib/x86_64-linux-gnu/odbc/libtdsodbc.so\n\
Setup = /usr/lib/x86_64-linux-gnu/odbc/libtdsS.so" >> /etc/odbcinst.ini

ENV NLTK_DATA /usr/share/nltk_data

RUN pip install -U nltk
RUN python -m nltk.downloader -d /usr/share/nltk_data stopwords

COPY --from=build /usr/local/bin/ /usr/local/bin/
COPY --from=build /usr/local/lib/ /usr/local/lib/

WORKDIR /code
COPY . /code

ENTRYPOINT python api.py

EXPOSE 5000
