import requests 

data = {"input": "Given an array `nums` of n integers and an integer `target`, find three integers in `nums` such that the sum is closest to `target`. Return the sum of the three integers. You may assume that each input would have exactly one solution."}
headers = {"Content-Type": "application/json"}

res = requests.get("http://127.0.0.1:3000/classify", json=data, headers=headers)

print("Classification results:", res.json())

res_input = requests.get("http://127.0.0.1:3000/classify/input", json=data, headers=headers)
print("Input classification:", res_input.json())

res_output = requests.get("http://127.0.0.1:3000/classify/output", json=data, headers=headers)
print("Output classification:", res_output.json())


res_topic = requests.get("http://127.0.0.1:3000/classify/topic", json=data, headers=headers)
print("Topic classification:", res_topic.json())

res_difficulty = requests.get("http://127.0.0.1:3000/classify/difficulty", json=data, headers=headers)
print("Difficulty classification:", res_difficulty.json())

res_keywords = requests.get("http://127.0.0.1:3000/classify/keywords", json=data, headers=headers)
print("Keywords extraction:", res_keywords.json())

# Define the user input dictionary
user_input = {
    "Topic": "dynamic programming",
    "Keywords": "array, search",
    "Input": "array of integers and target",
    "Output": "index of the integer number",
    "Difficulty": "Easy",
    "DataNotInteract": "1",
    "CountingRelated": "0",
    "FewKManyV": "1",
    "JoinData": "0",
    "ConditionProblem": "1"
}

# Make a POST request to the API endpoint
res = requests.get("http://127.0.0.1:3000/find_similar_problems", json={"input": user_input})

# Print the response
print("Similar problem IDs:", res.json()["similar_problem_ids"])
# print("Response content:", res.content)
