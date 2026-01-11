from openai import OpenAI

client = OpenAI(
    base_url="your base url",
    api_key="your api key"
)

def generate_response(query, context):
    system_prompt = """You are a helpful AI assistant. Answer the user's question based on the provided context. 
    If the context doesn't contain relevant information to answer the question fully, acknowledge this limitation."""

    user_prompt = f"""Context:
    {context}

    Question: {query}

    Please answer the question based on the provided context."""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )
    return response.choices[0].message.content