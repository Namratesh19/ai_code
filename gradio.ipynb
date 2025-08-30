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
        Updated history and empty string for the input box
    """
    if not message.strip():
        return history, ""
    
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
    
    # Add the conversation to history
    history.append([message, bot_response])
    
    return history, ""

# Create the Gradio interface
with gr.Blocks(title="LangGraph Chatbot") as demo:
    gr.Markdown("# LangGraph Chatbot")
    gr.Markdown("Chat with your AI assistant powered by LangGraph!")
    
    chatbot_interface = gr.Chatbot(
        label="Conversation",
        height=500,
        show_label=True
    )
    
    with gr.Row():
        msg_input = gr.Textbox(
            placeholder="Type your message here...",
            label="Message",
            scale=4
        )
        send_btn = gr.Button("Send", scale=1)
    
    # Handle message submission
    def submit_message(message, history):
        return chat_with_bot(message, history)
    
    # Connect the interface elements
    send_btn.click(
        fn=submit_message,
        inputs=[msg_input, chatbot_interface],
        outputs=[chatbot_interface, msg_input]
    )
    
    msg_input.submit(
        fn=submit_message,
        inputs=[msg_input, chatbot_interface],
        outputs=[chatbot_interface, msg_input]
    )
    
    # Add a clear button
    clear_btn = gr.Button("Clear Conversation")
    clear_btn.click(
        fn=lambda: ([], ""),
        outputs=[chatbot_interface, msg_input]
    )

if __name__ == "__main__":
    demo.launch(share=True)
