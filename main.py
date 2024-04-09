from flask import Flask, jsonify
from flask_restful import Api, Resource, reqparse
import os
from classifier import classify_all, predict_input, predict_output, predict_topic, predict_difficulty, extract_keywords
from similarity  import find_similar_problems


app = Flask(__name__)
api = Api(app)


class WelcomeRailway(Resource):
    def get(self):

        response = "Hello from ml-server!"

        return jsonify({"message": response}), 200
 
class Classification(Resource):
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument("input", type=str, required=True, help="Input text is required")
        args = parser.parse_args()
        
        user_input = args["input"]

        predictions = classify_all(user_input)

        return predictions, 200

class ClassifyInput(Resource):
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument("input", type=str, required=True, help="Input text is required")
        args = parser.parse_args()
        
        user_input = args["input"]

        prediction = predict_input(user_input)

        return {"input": prediction}, 200

class ClassifyOutput(Resource):
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument("input", type=str, required=True, help="Input text is required")
        args = parser.parse_args()
        
        user_input = args["input"]

        prediction = predict_output(user_input)

        return {"output": prediction}, 200

class ClassifyTopic(Resource):
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument("input", type=str, required=True, help="Input text is required")
        args = parser.parse_args()
        
        user_input = args["input"]

       
        prediction = predict_topic(user_input)

        return {"topic": prediction}, 200

class ClassifyDifficulty(Resource):
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument("input", type=str, required=True, help="Input text is required")
        args = parser.parse_args()
        
        user_input = args["input"]

        prediction = predict_difficulty(user_input)

        return {"difficulty": prediction}, 200

class ExtractKeywords(Resource):
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument("input", type=str, required=True, help="Input text is required")
        args = parser.parse_args()
        
        user_input = args["input"]

        # Extract keywords
        prediction = extract_keywords(user_input)

        return {"keywords": prediction}, 200

class FindSimilarProblems(Resource):
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument("input", type=dict, required=True, help="Input data is required")
        args = parser.parse_args()
        
        user_input = args["input"]

        # Find similar problems
        similar_problem_ids = find_similar_problems(user_input)

        return {"similar_problem_ids": similar_problem_ids}, 200


api.add_resource(FindSimilarProblems, "/find_similar_problems")
api.add_resource(ClassifyInput, "/classify/input")
api.add_resource(ClassifyOutput, "/classify/output")
api.add_resource(ClassifyTopic, "/classify/topic")
api.add_resource(ClassifyDifficulty, "/classify/difficulty")
api.add_resource(ExtractKeywords, "/classify/keywords")
api.add_resource(Classification, "/classify")   
api.add_resource(WelcomeRailway, "/")


if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))
