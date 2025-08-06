"""
Agentic RAG System with GPT-OSS via Hugging Face and Streamlit UI
"""

import streamlit as st
import os
import tempfile
from typing import Literal, List, Dict, Any
import json
from pathlib import Path

# Core dependencies
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain.tools.retriever import create_retriever_tool
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from openai import OpenAI


class AgenticRAGSystem:
    def __init__(self, hf_token: str, openai_api_key: str = None, model_name: str = "openai/gpt-oss-120b:cerebras"):
        """Initialize the Agentic RAG System with GPT-OSS"""
        self.hf_token = hf_token
        self.openai_api_key = openai_api_key
        self.model_name = model_name
        
        # Initialize OpenAI client for GPT-OSS via Hugging Face
        self.client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=hf_token,
        )
        
        # Initialize OpenAI embeddings
        if openai_api_key:
            self.embeddings = OpenAIEmbeddings(
                openai_api_key=openai_api_key
            )
        else:
            # Use OpenAI embeddings via Hugging Face as fallback
            self.embeddings = OpenAIEmbeddings(
                openai_api_key=hf_token,
                openai_api_base="https://router.huggingface.co/v1"
            )
        
        self.vectorstore = None
        self.retriever_tool = None
        self.graph = None
    
    def load_documents(self, pdf_files: List[str]) -> List:
        """Load and process PDF documents"""
        documents = []
        for pdf_file in pdf_files:
            loader = PyPDFLoader(pdf_file)
            documents.extend(loader.load())
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=500, chunk_overlap=100  # Increased chunk size for better context
        )
        
        doc_splits = text_splitter.split_documents(documents)
        
        # Create vector store
        self.vectorstore = InMemoryVectorStore.from_documents(
            documents=doc_splits, 
            embedding=self.embeddings
        )
        
        # Create retriever tool
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
        self.retriever_tool = create_retriever_tool(
            retriever, 
            'retrieval_docs', 
            'Search and return information from the uploaded documents'
        )
        
        return doc_splits
    
    def gpt_oss_completion(self, messages: List[Dict], tools: List = None) -> Dict:
        """Make completion request to GPT-OSS via Hugging Face"""
        try:
            formatted_messages = []
            for msg in messages:
                if isinstance(msg, BaseMessage):
                    role = msg.type
                    if role == "human":
                        role = "user"
                    elif role == "ai":
                        role = "assistant"
                    
                    message_dict = {"role": role, "content": str(msg.content or "")}
                    if hasattr(msg, "tool_call_id"):
                        message_dict["tool_call_id"] = msg.tool_call_id
                    formatted_messages.append(message_dict)
                else:
                    formatted_messages.append(msg)

            kwargs = {
                "model": self.model_name,
                "messages": formatted_messages,
                "temperature": 0.1,
                "max_tokens": 1000
            }
            
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"
            
            response = self.client.chat.completions.create(**kwargs)
            return {
                "content": response.choices[0].message.content,
                "tool_calls": getattr(response.choices[0].message, 'tool_calls', None),
                "role": "assistant"
            }
        except Exception as e:
            st.error(f"Error in GPT-OSS completion: {str(e)}")
            return {"content": f"Error: {str(e)}", "role": "assistant"}
    
    def generate_query_or_respond(self, state: MessagesState):
        """Generate response or decide to retrieve documents"""
        if not self.retriever_tool:
            # No documents loaded, respond directly
            response = self.gpt_oss_completion(state["messages"])
            return {"messages": [response]}
        
        # Convert retriever tool to OpenAI format
        tools = [{
            "type": "function",
            "function": {
                "name": "retrieval_docs",
                "description": "Search and return information from the uploaded documents",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query"
                        }
                    },
                    "required": ["query"]
                }
            }
        }]
        
        response = self.gpt_oss_completion(state["messages"], tools)
        
        return {"messages": [response]}
    
    class GradeDocuments(BaseModel):
        """Grade documents for relevance"""
        binary_score: str = Field(
            description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
        )
    
    def grade_documents(self, state: MessagesState) -> Literal["generate_answer", "rewrite_question"]:
        """Grade retrieved documents for relevance"""
        question = state["messages"][0].content
        print('Question', question)
        context = state["messages"][-1].content
        print('Context', context)
        
        grade_prompt = f"""You are a grader assessing relevance of a retrieved document to a user question.
        
        Here is the retrieved document:
        {context}
        
        Here is the user question: {question}
        
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant.
        Respond with a JSON object containing only a 'binary_score' field with value 'yes' or 'no'.
        """
        
        messages = [{"role": "user", "content": grade_prompt}]
        response = self.gpt_oss_completion(messages)
        
        try:
            # Try to extract binary score from response
            content = response["content"].strip()
            if "yes" in content.lower():
                return "generate_answer"
            else:
                return "rewrite_question"
        except:
            # Default to generating answer if parsing fails
            return "generate_answer"
    
    def rewrite_question(self, state: MessagesState):
        """Rewrite the question for better retrieval"""
        question = state["messages"][0].content
        print("Question", question)
        
        rewrite_prompt = f"""Look at the input and try to reason about the underlying semantic intent/meaning.
        
        Here is the initial question:
        {question}
        
        Formulate an improved question that would be better for document search:
        """
        
        messages = [{"role": "user", "content": rewrite_prompt}]
        response = self.gpt_oss_completion(messages)
        
        return {"messages": [{"role": "user", "content": response["content"]}]}
    
    def generate_answer(self, state: MessagesState):
        """Generate final answer using retrieved context"""
        question = state["messages"][0].content
        print('Question', question)
        context = state["messages"][-1].content
        print('Context', context)
        
        answer_prompt = f"""You are an assistant for question-answering tasks.
        Use the following pieces of retrieved context to answer the question.
        If you don't know the answer, just say that you don't know.
        Use three sentences maximum and keep the answer concise.
        
        Question: {question}
        Context: {context}
        """
        
        messages = [{"role": "user", "content": answer_prompt}]
        response = self.gpt_oss_completion(messages)
        
        return {"messages": [response]}
    
    def build_graph(self):
        """Build the LangGraph workflow"""
        workflow = StateGraph(MessagesState)
        
        # Add nodes
        workflow.add_node("generate_query_or_respond", self.generate_query_or_respond)
        workflow.add_node("retrieve", ToolNode([self.retriever_tool]) if self.retriever_tool else lambda x: x)
        workflow.add_node("rewrite_question", self.rewrite_question)
        workflow.add_node("generate_answer", self.generate_answer)
        
        # Add edges
        workflow.add_edge(START, "generate_query_or_respond")
        
        # Conditional edges for retrieval decision
        workflow.add_conditional_edges(
            "generate_query_or_respond",
            tools_condition,
            {
                "tools": "retrieve",
                END: END,
            },
        )
        
        # Conditional edges after retrieval
        workflow.add_conditional_edges(
            "retrieve",
            self.grade_documents,
        )
        
        workflow.add_edge("generate_answer", END)
        workflow.add_edge("rewrite_question", "generate_query_or_respond")
        
        # Compile the graph
        self.graph = workflow.compile()
        
        return self.graph
    
    def process_query(self, query: str) -> str:
        """Process a query through the agentic RAG system"""
        if not self.graph:
            self.build_graph()
        
        try:
            result = self.graph.invoke({
                "messages": [{"role": "user", "content": query}]
            })
            
            return result["messages"][-1].content
        except Exception as e:
            return f"Error processing query: {str(e)}"


# Streamlit UI
def main():
    st.set_page_config(
        page_title="Agentic RAG with GPT-OSS", 
        page_icon="🤖", 
        layout="wide"
    )
    
    st.title("🤖 Agentic RAG System with GPT-OSS")
    st.markdown("Upload PDFs and ask questions using OpenAI's GPT-OSS models via Hugging Face!")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Hugging Face token input
        hf_token = st.text_input(
            "Hugging Face Token", 
            type="password",
            help="Get your token from https://huggingface.co/settings/tokens"
        )
        
        # OpenAI API key input
        openai_api_key = st.text_input(
            "OpenAI API Key", 
            type="password",
            help="For better embeddings. If not provided, will use HF embeddings"
        )
        
        # Show selected model
        st.info("🤖 **Model**: GPT-OSS 120B (Cerebras)")
        
        st.markdown("---")
        st.markdown("### 📖 About")
        st.markdown("""
        This system uses:
        - **GPT-OSS 120B**: OpenAI's 120B parameter model
        - **Cerebras**: High-performance inference provider  
        - **OpenAI Embeddings**: For document retrieval
        - **LangGraph**: Agentic workflows
        - **RAG**: Retrieval-augmented generation
        """)
        
        if not openai_api_key:
            st.warning("💡 **Tip**: Add OpenAI API key for better embeddings quality")

    
    # Initialize session state
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = None
    if "documents_loaded" not in st.session_state:
        st.session_state.documents_loaded = False
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Initialize RAG system when tokens are provided
    if hf_token and not st.session_state.rag_system:
        try:
            st.session_state.rag_system = AgenticRAGSystem(
                hf_token=hf_token, 
                openai_api_key=openai_api_key,
                model_name="openai/gpt-oss-120b:cerebras"
            )
            st.success("✅ RAG System initialized with GPT-OSS 120B (Cerebras)!")
        except Exception as e:
            st.error(f"❌ Error initializing RAG system: {str(e)}")
    
    if not hf_token:
        st.warning("⚠️ Please enter your Hugging Face token in the sidebar to get started.")
        return
    
    # File upload section
    st.header("📄 Document Upload")
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type="pdf",
        accept_multiple_files=True,
        help="Upload PDF documents to create a knowledge base"
    )
    
    if uploaded_files and st.session_state.rag_system:
        if st.button("🔄 Process Documents"):
            with st.spinner("Processing documents..."):
                try:
                    # Save uploaded files temporarily
                    temp_files = []
                    for uploaded_file in uploaded_files:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                            tmp.write(uploaded_file.getvalue())
                            temp_files.append(tmp.name)
                    
                    # Load documents into RAG system
                    doc_splits = st.session_state.rag_system.load_documents(temp_files)
                    st.session_state.documents_loaded = True
                    
                    # Clean up temp files
                    for temp_file in temp_files:
                        os.unlink(temp_file)
                    
                    st.success(f"✅ Processed {len(uploaded_files)} documents into {len(doc_splits)} chunks!")
                    
                except Exception as e:
                    st.error(f"❌ Error processing documents: {str(e)}")
    
    # Chat interface
    st.header("💬 Chat Interface")
    
    if not st.session_state.rag_system:
        st.info("🔧 Please configure your Hugging Face token first.")
        return
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if query := st.chat_input("Ask a question..."):
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": query})
        
        with st.chat_message("user"):
            st.write(query)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.rag_system.process_query(query)
                    st.write(response)
                    
                    # Add assistant response to history
                    st.session_state.chat_history.append({
                        "role": "assistant", 
                        "content": response
                    })
                    
                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    st.write(error_msg)
                    st.session_state.chat_history.append({
                        "role": "assistant", 
                        "content": error_msg
                    })
    
    # Clear chat history
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()
    
    # Instructions
    with st.expander("📋 How to Use"):
        st.markdown("""
        1. **Get HF Token**: Get your Hugging Face token from [settings page](https://huggingface.co/settings/tokens)
        2. **Enter Token**: Paste it in the sidebar
        3. **Upload PDFs**: Upload documents to create knowledge base
        4. **Ask Questions**: Chat with the AI assistant
        
        **Features:**
        - 🎯 **Smart Routing**: Decides whether to search documents or respond directly
        - 📊 **Document Grading**: Evaluates relevance of retrieved content
        - 🔄 **Query Rewriting**: Improves search queries automatically
        - 🧠 **GPT-OSS 120B**: Uses OpenAI's powerful 120B parameter model via Cerebras
        """)


if __name__ == "__main__":
    main()