from flask import Flask, jsonify, request

import torch
import torch.nn.functional as F

from . import db
from ..ml.models import SentimentClassifier
from ..ml.data import to_feature_vector, id_to_rating

app = Flask(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True

seq_len = 250
all_chars = 'abcdefghijklmnopqrstuvwxyz0123456789!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ '

weights_path = "../checkpoints/trained_weights.pth"
model = SentimentClassifier(feature_dim=len(all_chars),
                            seq_len=seq_len,
                            conv_num_kernels=[128, 256, 512, 1024, 512, 256],
                            conv_kernel_sizes=[7, 7, 5, 5, 5, 3],
                            pool_sizes=[3, 3, None, None, None, 3],
                            batchnorm=True)
# model = SentimentClassifier(feature_dim=len(all_chars),
#                             seq_len=seq_len)
if torch.cuda.is_available():
    weights = torch.load(weights_path)
else:
    weights = torch.load(weights_path, map_location="cpu")
model.load_state_dict(weights)
model.eval()


@app.route("/", methods=['GET'])
def home():
    return "server is running"


@app.route("/review", methods=['GET'])
def get_reviews():
    query = db.Review.select()
    return jsonify([r.serialize() for r in query])


@app.route("/review", methods=['POST'])
def post_review():
    required_fields = [
        "review",
        "rating",
        "suggested_rating",
        "sentiment_score",
        "brand",
        "user_agent",
        "ip_address",
    ]
    data = request.get_json()
    for k in required_fields:
        if k not in data.keys():
            return jsonify({"error": f"'{k}' not found in body"})
    query = db.Review.create(**data)
    return jsonify(query.serialize())


@app.route("/predict", methods=['POST'])
def predict():
    data = request.get_json()
    if "review" not in data:
        return jsonify({"error": "'review' not found in body"})
    else:
        rating, conf = predict_wrapper(data['review'])
        print(f"predictions: {rating}, {conf}")
        score = compute_dummy_score(rating, conf)
        return jsonify({"score": score})


def compute_dummy_score(rating, conf):
    shift = {
        "good": 2/3,
        "average": 1/3,
        "bad": 0,
    }
    return shift[rating] + (1/3) * conf if rating != "bad" \
        else shift[rating] + (1/3) * (1 - conf)


def predict_wrapper(review):
    x = to_feature_vector(review, all_chars, seq_len)
    x = x.unsqueeze(0)
    with torch.no_grad():
        scores = F.softmax(model(x), dim=1).squeeze()
        conf, label_id = torch.max(scores, 0)
    return id_to_rating(label_id.item()), conf.item()
