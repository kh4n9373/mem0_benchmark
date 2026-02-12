
from openai import OpenAI
import time

print("Initializing OpenAI client...")
client = OpenAI(
    base_url="http://localhost:8002/v1",
    api_key="dummy",
    timeout=10.0
)

print("Sending request to vLLM...")
try:
    start_time = time.time()
    response = client.chat.completions.create(
        model="Qwen/Qwen3-8B",
        messages=[
            {"role": "user", "content": "Hello, are you working?"}
        ],
        max_tokens=50
    )
    duration = time.time() - start_time
    print(f"\n✅ Success! Time: {duration:.2f}s")
    print(f"Response: {response.choices[0].message.content}")
except Exception as e:
    print(f"\n❌ Error: {e}")
