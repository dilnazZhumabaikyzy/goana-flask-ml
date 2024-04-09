import joblib
import pandas as pd
directory = ""

model_input = joblib.load(directory + "input_classifier.joblib")
model_output =  joblib.load(directory + "output_classifier.joblib")
model_difficulty = joblib.load(directory + "difficulty_classifier.joblib")
model_topics = joblib.load(directory + "topics_classifier.pkl")
keywords_extractor = joblib.load(directory + "keywords_extractor.pkl")
keywords_decode = joblib.load(directory + "keywords_decode.pkl")

tfidf_vectorizer = joblib.load(directory + "tfidf_vectorizer_beta.joblib")
tfidf_vectorizer_sigma = joblib.load(directory + "sigma_vectorizer.pkl")


def predict_input(user_input):
    user_input_vectorized = tfidf_vectorizer.transform([user_input])
    prediction = model_input.predict(user_input_vectorized)[0]

    return prediction

def predict_output(user_input):
    user_input_vectorized = tfidf_vectorizer.transform([user_input])
    prediction = model_output.predict(user_input_vectorized)[0]

    return prediction

def predict_topic(user_input):
    user_input_vectorized = tfidf_vectorizer_sigma.transform([user_input])
    prediction = model_topics.predict(user_input_vectorized)[0]

    return prediction


def predict_difficulty(user_input):
    user_input_vectorized = tfidf_vectorizer.transform([user_input])
    prediction = model_difficulty.predict(user_input_vectorized)[0]

    return prediction

def extract_keywords(user_input):
    user_input_vectorized = tfidf_vectorizer.transform([user_input])
    predicted_keywords = keywords_decode.inverse_transform(keywords_extractor.predict(user_input_vectorized))
    

    return predicted_keywords

def classify_all(user_input):  
    predictions = {}

    predictions['input'] = predict_input(user_input)
    predictions['output'] = predict_output(user_input)
    predictions['topic'] = predict_topic(user_input)
    predictions['difficulty'] = predict_difficulty(user_input)
    predictions['keywords'] = extract_keywords(user_input)

    return predictions


# user_input = "Given an array `nums` of n integers and an integer `target`, find three integers in `nums` such that the sum is closest to `target`. Return the sum of the three integers. You may assume that each input would have exactly one solution."

# print(predict_input(user_input))
# print(predict_output(user_input))
# print(predict_difficulty(user_input))
# print(predict_topic(user_input))
# print(classify_all(user_input))