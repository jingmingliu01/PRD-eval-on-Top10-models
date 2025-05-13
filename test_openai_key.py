from together import Together

client = Together()

response = client.chat.completions.create(
    model="Qwen/Qwen3-235B-A22B-fp8-tput",
    messages=[
      {
        "role": "user",
        "content": "Hello, how are you?"
      }
    ],
    stream=True
)
for token in response:
    if hasattr(token, 'choices'):
        print(token.choices[0].delta.content, end='', flush=True)
