import os
import torch
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.llms.base import LLM
from typing import Optional, List, Any

# Environment Setup (Using GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Set Seed for Reproducibility
SEED = 42
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# Load and process documents
references_dir = "../References/"
file_names = [f for f in os.listdir(references_dir) if os.path.isfile(os.path.join(references_dir, f))]
pdf_files = [os.path.join(references_dir, file) for file in file_names]

def load_documents(pdf_paths):
    documents = []
    for pdf in pdf_paths:
        loader = PyPDFLoader(pdf)
        documents.extend(loader.load())
    return documents

documents = load_documents(pdf_files)

# Splitting documents for embeddings
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_documents = text_splitter.split_documents(documents)

# Embeddings and Vector Store
embedding_model = HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-MiniLM-L6-v2',
    model_kwargs={"device": device}
)
vector_store = FAISS.from_documents(split_documents, embedding_model)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# Define Prompt
prompt_template = """
Answer the following question very shortly(no more than 50 words) based solely on the provided context.
If the answer is not present in the context, say 'I'm sorry, but I don't have enough information to answer that.'

Context:
{context}

Question:
{input}

Answer:
""".strip()

prompt = ChatPromptTemplate.from_template(template=prompt_template)

# Generator Model Setup
model_name = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, device_map="auto", torch_dtype=torch.float16
)

llm_pipeline = pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    do_sample=True,
    temperature=0.3,
    repetition_penalty=1.2
)

# Custom wrapper for LangChain compatibility
class CustomPipelineLLM(LLM):
    pipeline: Any

    @property
    def _llm_type(self) -> str:
        return "custom_pipeline"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        output = self.pipeline(prompt, max_new_tokens=256, do_sample=True, temperature=0.3)[0]['generated_text']
        return output.strip().split("Answer:")[-1].strip()

llm_custom = CustomPipelineLLM(pipeline=llm_pipeline)

# Create RAG Chain
document_chain = create_stuff_documents_chain(llm_custom, prompt)
rag_chain = create_retrieval_chain(retriever, document_chain)
