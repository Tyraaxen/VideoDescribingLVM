import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Initialize OpenAI client with the API key
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")  # Ensure the .env file contains OPENAI_API_KEY
)

def chat_with_gpt(user_input, model="gpt-4"):
    """
    Sends a user input to the OpenAI API and retrieves the assistant's response.
    """
    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_input},
            ],
            model=model,
        )
        # Extract and return the assistant's reply
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"An error occurred: {e}"

if __name__ == "__main__":
    print("Welcome to your chatbot! Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        # Get the chatbot's response
        bot_response = chat_with_gpt(user_input)
        print(f"Bot: {bot_response}")
