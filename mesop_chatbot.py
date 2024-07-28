import mesop as me
import mesop.labs as mel
import subprocess

@me.page(
    security_policy=me.SecurityPolicy(
        allowed_iframe_parents=["https://google.github.io"]
    ),
    path="/chat",
    title="Mesop Demo Chat",
)
def chat_page():
    mel.chat(transform, title="Mesop Demo Chat", bot_user="Mesop Bot")

@me.page(path="/")
def home():
    me.text("Welcome to the Mesop Chatbot!")
    me.text("Please navigate to the /chat path to use the chat interface.")

def transform(input: str, history: list[mel.ChatMessage]):
    yield "Processing your request..."
    response = run_cli_command(input)
    yield response

def run_cli_command(query):
    try:
        result = subprocess.run(
            [
                "python", "-m", "graphrag.query",
                "--root", "./ragtest",
                "--method", "global",
                query
            ],
            capture_output=True, text=True
        )
        return result.stdout.strip()  # Strip to remove any extraneous whitespace
    except Exception as e:
        return str(e)

if __name__ == '__main__':
    me.run()
