import os
os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
os.environ['GRPC_ENABLE_FORK_SUPPORT'] = '0'
import google.generativeai as genai
import sys

keys = [
    "AIzaSyDSj5KqM04bEFLMeb_0p7QGJ0hSGKKP1vY",
    "AIzaSyBrL21YO9OXZpJ5KX8D8EApuwvcBjEI8XE",
    "AIzaSyATukiIJYPqwQ9IdirGEzQdzp_iEXq_6rA",
    "AIzaSyBo6amqWHwFwPq1-SKtaDLWggOxhGB2_Bs"
]

for i, key in enumerate(keys):
    print(f"Testing Key {i}...")
    try:
        genai.configure(api_key=key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content("Hello, this is a test. Reply with 'OK'.")
        print(f"Key {i} Response: {response.text}")
    except Exception as e:
        print(f"Key {i} Failed: {e}")
