import openai
import os

def get_java_implementation(user_input):
    openai.api_key = os.getenv('OPENAI_API_KEY')  

    if not openai.api_key:
        raise ValueError("OpenAI API key is not set. Please set the environment variable 'OPENAI_API_KEY'.")

    model = "gpt-3.5-turbo"

    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "Generate only the Java 8 code implementation based on the problem description provided by the user. Do not include any explanatory text or comments."},
            {"role": "user", "content": user_input}
        ]
    )
    return response.choices[0].message['content']