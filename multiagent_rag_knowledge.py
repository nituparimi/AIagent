# multiagent rag with external db integration
# pre-reqs
# Docker Desktop installed
# Python environment
# Groq API key
# Basic understanding of Vector databases

# Project Overview
# Read PDF documents from URLs
# Store content in a Vector database (PG Vector)
# Create an AI assistant to interact with the stored knowledge
# Provide accurate responses based on the document content

# requirements.txt
sqlalchemy
pgvector
psycopg2-binary
pypdf

docker run --name pgvector \
    -e POSTGRES_PASSWORD=postgres \
    -p 5432:5432 \
    -d \
    pgvector/pgvector:pg16

#Initial imports
from phi.assistant import Assistant
from phi.storage.assistant.postgres import PGAssistantStorage
from phi.knowledge.pdf import PDFURLKnowledgeBase
from phi.vectordb.pgvector import PGVector2
from dotenv import load_dotenv
import os

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

#DB configuration
DB_URL = "postgresql://postgres:postgres@localhost:5432/postgres"

# Knowledgebase setup

knowledge_base = PDFURLKnowledgeBase(
    urls=["https://phi-public.s3.amazonaws.com/recipes/thai_recipes.pdf"],
    vector_db=PGVector2(
        collection_name="recipes",
        db_url=DB_URL
    )
)

knowledge_base.load(
    storage=PGAssistantStorage(
        table_name="pdf_assistant",
        db_url=DB_URL
    )
)

#Creating pdf assistant
def pdf_assistant(
    new: bool = False,
    user: str = "user"
) -> None:
    assistant = Assistant(
        run_id="pdf_assistant",
        user_id=user,
        knowledge_base=knowledge_base,
        storage=PGAssistantStorage(
            table_name="pdf_assistant",
            db_url=DB_URL
        ),
        show_tool_calls=True,
        search_knowledge=True,
        read_chat_history=True
    )

    if new or not assistant.run_id:
        assistant.run_id = "pdf_assistant"
    
    assistant.start()
    assistant.cli(markdown=True)
    
# Running the Application
if __name__ == "__main__":
    import typer
    typer.run(pdf_assistant)

# Assistant can handle queries like
# "List all the dishes in the document"
# "What are the ingredients for Masaman Gai?"
# "How to prepare this dish?"

# Features
# Vector Database Integration: Uses PG Vector for efficient storage and retrieval
# PDF Processing: Automatically extracts and vectorizes PDF content
# Chat History: Maintains conversation context
# Tool Integration: Shows tool calls in responses

# common issues
# Docker Issues: Ensure Docker Desktop is running before starting
# Library Dependencies: Install all required packages from requirements.txt
# Database Connection: Verify PostgreSQL connection string
# PDF Processing: Ensure PDF URLs are accessible


#This project demonstrates how to build a sophisticated AI assistant that can interact with vector databases and process PDF documents. 
#It showcases the power of combining multiple tools and technologies to create complex workflows that solve real-world problems.
