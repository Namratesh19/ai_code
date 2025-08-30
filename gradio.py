import gradio as gr
from langgraph_backend import chatbot
from langchain_core.messages import HumanMessage

CONFIG = {'configurable': {'thread_id': 'thread-1'}}

def chat_with_bot(message, history):
    """
    Handle chat interaction with the LangGraph chatbot
    
    Args:
        message: Current user message
        history: List of [user_msg, bot_msg] pairs from previous conversation
    
    Returns:
        Bot response string
    """
    if not message.strip():
        return ""
    
    # Stream the response from the chatbot
    response_chunks = []
    for message_chunk, metadata in chatbot.stream(
        {'messages': [HumanMessage(content=message)]},
        config=CONFIG,
        stream_mode='messages'
    ):
        if hasattr(message_chunk, 'content') and message_chunk.content:
            response_chunks.append(message_chunk.content)
    
    # Combine all chunks into final response
    bot_response = ''.join(response_chunks)
    
    return bot_response

def launch():
    gr.ChatInterface(
        chat_with_bot,
        title="LangGraph Chatbot",
        description="Chat with your AI assistant powered by LangGraph!"
    ).launch(share=True)

if __name__ == "__main__":
    launch()
