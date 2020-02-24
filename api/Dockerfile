FROM python:3.6

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt
    #  apk add --no-cache postgresql-libs && \
    #  apk add --no-cache --virtual .build-deps gcc musl-dev postgresql-dev && \
#  apk --purge del .build-deps

COPY . .

CMD ["gunicorn", "-b", "0.0.0.0:5000", "--workers", "4", "app:app"]