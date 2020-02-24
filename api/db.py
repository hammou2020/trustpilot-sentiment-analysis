import peewee as pw
import config


db = pw.PostgresqlDatabase(config.POSTGRES_DB,
                           user=config.POSTGRES_USER,
                           password=config.POSTGRES_PASSWORD,
                           host=config.POSTGRES_HOST,
                           port=config.POSTGRES_PORT)


class Review(pw.Model):
    review = pw.TextField()
    rating = pw.IntegerField()
    suggested_rating = pw.IntegerField()
    sentiment_score = pw.FloatField()
    brand = pw.TextField()
    user_agent = pw.TextField()
    ip_address = pw.TextField()

    def serialize(self):
        return {
            'id': self.id,
            'review': self.review,
            'rating': int(self.rating),
            'suggested_rating': int(self.suggested_rating),
            'sentiment_score': self.sentiment_score,
            'brand': self.brand,
            'user_agent': self.user_agent,
            'ip_address': self.ip_address,
        }

    class Meta:
        database = db


db.connect()
db.create_tables([Review])
