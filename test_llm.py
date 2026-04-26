import os
from dotenv import load_dotenv
from backend.llm import GeminiEngine

load_dotenv()

def test():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not set in environment.")
        return
        
    engine = GeminiEngine(api_key=api_key)
    
    if not engine.is_ready:
        print("Engine not ready. Is google-genai installed?")
        return
        
    print("Testing analyze_thought...")
    text = "If I don't get this perfect, the entire project is ruined."
    result = engine.analyze_thought(text)
    print("Result:")
    import json
    print(json.dumps(result, indent=2))
    
    bias = result.get('bias', 'Catastrophizing') if result else 'Catastrophizing'
    print("\nTesting generate_chat_reply...")
    chat_reply = engine.generate_chat_reply(text, bias, [])
    print("Chat Reply:")
    print(json.dumps(chat_reply, indent=2))

if __name__ == "__main__":
    test()
