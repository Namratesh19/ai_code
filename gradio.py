import gradio as gr
from langgraph_backend import chatbot
from langchain_core.messages import HumanMessage

CONFIG = {'configurable': {'thread_id': 'thread-1'}}

def chat_with_bot(message, history):
    """
    Handle chat interaction with the LangGraph chatbot
    
    Args:
        message: Current user message (string)
        history: List of message dictionaries with 'role' and 'content' keys
    
    Yields:
        Accumulated bot response tokens
    """
    print(f"Function called! Message: '{message}'")  # Debug line
    print(f"History: {history}")  # Debug line
    
    if not message or not message.strip():
        yield "Please enter a message."
        return
    
    try:
        # Use invoke to get the response
        result = chatbot.invoke(
            {'messages': [HumanMessage(content=message.strip())]},
            config=CONFIG
        )
        
        # Extract the AI response from the result
        ai_response = ""
        if 'messages' in result and result['messages']:
            last_message = result['messages'][-1]
            if hasattr(last_message, 'content'):
                ai_response = last_message.content
        
        if not ai_response:
            yield "Sorry, I couldn't generate a response. Please try again."
            return
        
        # Simulate streaming by yielding character by character
        output = ""
        for char in ai_response:
            output += char
            yield output
        
    except Exception as e:
        print(f"Error in chat_with_bot: {e}")  # Debug line
        yield f"Error: {str(e)}. Please check your API configuration and try again."

def launch():
    gr.ChatInterface(
        chat_with_bot,
        title="LangGraph Chatbot",
        description="Chat with your AI assistant powered by LangGraph!",
        autofocus=False,
        type="messages",  # This is crucial!
    ).queue().launch(
        share=True,
        height=850,
    )

if __name__ == "__main__":
    launch()
