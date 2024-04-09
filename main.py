from flask import Flask, jsonify
from flask_restful import Api, Resource, reqparse
import os

app = Flask(__name__)
api = Api(app)


class WelcomeRailway(Resource):
    def get(self):

        response = "Hello from ml-server!"

        return jsonify({"message": response}), 200
    
api.add_resource(WelcomeRailway, "/")

if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))
