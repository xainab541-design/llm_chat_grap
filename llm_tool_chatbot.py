import os
import requests
import logging
from typing import Optional
from dotenv import load_dotenv
from openai import OpenAI

# ------------------ LOGGING SETUP ------------------

logger = logging.getLogger("LLM_Chatbot")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

file_handler = logging.FileHandler("llm_chatbot.log")
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

# ------------------ LOAD ENV ------------------

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not GROQ_API_KEY or not TAVILY_API_KEY:
    logger.error("GROQ_API_KEY and TAVILY_API_KEY must be set in environment or .env file.")
    raise ValueError("Missing API keys.")

client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")

# ------------------ TAVILY SEARCH ------------------

def tavily_search(query: str, limit: int = 3) -> str:
    url = "https://api.tavily.com/search"
    headers = {"Authorization": f"Bearer {TAVILY_API_KEY}"}
    payload = {"query": query, "limit": limit}

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        results = response.json().get("results", [])

        if not results:
            return "No results found via Tavily."

        formatted_results = "\n".join(
            [
                f"{i+1}. {r.get('title','No title')}: {r.get('snippet','No snippet')} ({r.get('url','No URL')})"
                for i, r in enumerate(results)
            ]
        )

        return formatted_results

    except Exception as e:
        logger.exception(f"Error fetching results from Tavily: {str(e)}")
        return f"Error fetching results from Tavily: {str(e)}"

# ------------------ LLM FUNCTION ------------------

def ask_llm(question: str, context: Optional[str] = None) -> str:
    try:
        prompt = question
        if context:
            prompt = f"Use the following context to answer the question:\n{context}\n\nQuestion: {question}"

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are an expert AI assistant with deep knowledge across multiple domains. Provide accurate, detailed, and well-structured answers. Be clear, concise, and helpful. If you're unsure about something, say so."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )

        answer = response.choices[0].message.content
        logger.debug(f"Received from Groq: {answer}")
        return answer

    except Exception as e:
        logger.exception(f"Unexpected error in ask_llm (Groq): {e}")
        return "An unexpected error occurred while contacting Groq API."

# ------------------ DYNAMIC SEARCH DECISION ------------------

def needs_search(question: str) -> bool:
    """
    Ask the LLM whether this question requires real-time search.
    Returns True if Tavily search is needed.
    """
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are an intelligent assistant that determines if a user's question requires current, real-time information from the internet (news, stock prices, weather, latest events, recent statistics). Answer only 'Yes' or 'No'. Say 'Yes' if the question asks about recent events, current news, live data, or anything that changes frequently and needs up-to-date information. Say 'No' for general knowledge, definitions, how-to questions, or historical facts."},
                {"role": "user", "content": f"Question: {question}"}
            ],
            temperature=0
        )

        answer = response.choices[0].message.content.strip().lower()
        logger.debug(f"Search decision for '{question}': {answer}")
        return answer.startswith("yes")
    except Exception as e:
        logger.exception(f"Error in needs_search: {e}")
        # Default to Groq only if something fails
        return False

# ------------------ MAIN CHATBOT ------------------

def chatbot(query: str) -> dict:
    output = {"query": query, "source": "LLM only", "response": ""}

    try:
        if needs_search(query):
            output["source"] = "Tavily + Groq"
            tavily_results = tavily_search(query)
            answer = ask_llm(query, context=tavily_results)
            output["response"] = answer
            output["tavily_results"] = tavily_results
        else:
            output["response"] = ask_llm(query)

        logger.info(f"Query processed. Source: {output['source']}")
        return output

    except Exception as e:
        logger.exception(f"Error in chatbot function: {e}")
        output["response"] = "Sorry, something went wrong. Check logs for details."
        return output

# ------------------ MAIN LOOP ------------------

if __name__ == "__main__":
    print("Groq Smart Chatbot is ready! Type your questions. Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            user_input = input("You: ").strip()

            if user_input.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break

            result = chatbot(user_input)

            print("\n--- Chatbot Response ---")
            print(result["response"])

            if "tavily_results" in result:
                print("\n--- Tavily Results ---")
                print(result["tavily_results"])

            print("------------------------\n")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            logger.exception(f"Unexpected error in main loop: {e}")
            print("An unexpected error occurred. Check logs.")

