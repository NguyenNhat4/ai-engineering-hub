import os
import httpx
from dotenv import load_dotenv
from python_a2a import A2AServer, skill, agent, run_server, TaskStatus, TaskState

load_dotenv()

GEMINI_API_KEY = os.getenv("key")
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"


@agent(
    name="Gemini LLM Agent",
    description="Routes queries to the appropriate agent using Gemini LLM",
    version="1.0.0",
    url="http://localhost:8000"
)
class GeminiLLMAgent(A2AServer):

    def handle_task(self, task):
        input_message = task.message["content"]["text"]

        # Call Gemini API
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": GEMINI_API_KEY
        }

        body = {
            "contents": [{"role": "user", "parts": [{"text": input_message}]}],
            "generationConfig": {"temperature": 0.7}
        }

        with httpx.Client(timeout=60.0) as client:
            response = client.post(GEMINI_URL, json=body, headers=headers)
            response.raise_for_status()
            data = response.json()

        # Extract response text
        content = ""
        if data.get("candidates"):
            parts = data["candidates"][0].get("content", {}).get("parts", [])
            content = "".join(p.get("text", "") for p in parts)

        task.artifacts = [{
            "parts": [{"type": "text", "text": content}]
        }]
        task.status = TaskStatus(state=TaskState.COMPLETED)

        return task


if __name__ == "__main__":
    agent = GeminiLLMAgent(url="http://localhost:8000")
    run_server(agent, port=8000)
