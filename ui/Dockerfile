
FROM python:3.6

WORKDIR /ui

COPY requirements.txt .
RUN pip install -r requirements.txt
    #  apk add --no-cache postgresql-libs && \
    #  apk add --no-cache --virtual .build-deps gcc musl-dev postgresql-dev && \
#  apk --purge del .build-deps

COPY . .

EXPOSE 8050

CMD ["gunicorn", "-b", "0.0.0.0:8050", "app:server"]