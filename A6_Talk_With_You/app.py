import gradio as gr
from rag_module import rag_chain

# Function to process user input and generate response
def chatbot_response(user_input, history=[]):
    if not user_input.strip():
        return history, history  # Prevent empty submissions
    
    no_info_responses = [
        "I'm sorry, but I don't have enough information to answer that.",
        "I don’t know.",
        "The given context does not provide enough information.",
        "No relevant information found."
    ]
    
    result = rag_chain.invoke({"input": user_input})
    answer = result['answer'].strip()
    clean_answer = answer.split("\n")[0]
    sources = result.get('context', [])  # 'context' contains retrieved documents
    source_texts = list(set(doc.metadata.get('source', 'Unknown Source') for doc in sources if 'source' in doc.metadata))  # Ensure 'source' key exists
    if clean_answer in no_info_responses or "I'm sorry, but I don't have enough information to answer that." in clean_answer.lower():
        source_list = "No sources available."  # Do not display sources
    else:
        source_list = "\n".join([f"- {src}" for src in source_texts]) if sources else "No sources available."
    
    # formatted_response = f"**Response:**\n{clean_answer}\n\n**Sources:**\n{sources}"
    formatted_response = f"**Response:**\n{clean_answer}\n\n**Sources:**\n{source_list}"
    # formatted_response = f"**Response:**\n{result}"
    history.append((user_input, formatted_response))    
    return history, history  # Update chat history

# Gradio Interface with a Send button
chatbot = gr.Interface(
    fn=chatbot_response,
    inputs=[
        gr.Textbox(placeholder="Ask me anything...", label="Your Question"),
        gr.State([]),  # Chat history state
    ],
    outputs=[
        gr.Chatbot(label="Chat History"),
        gr.State([]),
    ],
    title="Let's Talk with Yourself - A6",
    description=(
        "**Assignment A6 - Developed by Zwe Htet (st125338)**\n\n"
        "This chatbot uses Retrieval-Augmented Generation (RAG) to "
        "answer questions about your personal information based on provided documents."
    ),
    theme="default",
    live=False,  # This prevents auto-send; requires a button press
    allow_flagging="never",
    examples=[
        ["Can you describe your current role or job responsibilities?"],
        ["What are your core beliefs regarding the role of technology in shaping society?"],
        ["How do you think cultural values should influence technological advancements?"],
    ],
)

# Launch the web app with a shareable link
if __name__ == "__main__":
    chatbot.launch()


