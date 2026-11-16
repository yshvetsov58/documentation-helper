import os
import random
import time
from typing import Any, Dict

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from pinecone import Pinecone, RateLimitError, DenseVectorQuery
from logger import log_success, log_error, log_warning
from langchain_openai import OpenAIEmbeddings
load_dotenv()
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
PINECONE_INDEX_NAME = "documentation-openai"
index = pc.index(PINECONE_INDEX_NAME)
model = init_chat_model("gpt-5.2", model_provider="openai")
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small", show_progress_bar=False, chunk_size=50, retry_min_seconds=30
)

@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve relevant documentation to answer questions about LangChain."""
    max_attempts = 5
    query_vector = embeddings.embed_query(query)
    for attempt in range(max_attempts):
        try:
            results = index.documents.search(
                top_k = 10,
                namespace = "langchain-docs-v2",
                score_by = [
                    DenseVectorQuery(
                        field = "embedding",
                        values = query_vector,
                    )
                ],
                include_fields = ["chunk_text", "source"],
                timeout = 60,
            )

        except RateLimitError as error:
            log_warning(f"Pinecone rate limit details: {error}")
            if attempt == max_attempts - 1:
                log_error(
                    f"Retrieval failed after {max_attempts} attempts: "
                    "rate limit exceeded."
                )
                raise

            backoff = min(10 * (2 ** attempt), 60)
            delay = max(backoff, error.retry_after or 0)
            delay += random.uniform(0, 3)

            log_warning(
                f"Rate limit exceeded. Retrying in {delay:.1f} seconds "
                f"(attempt {attempt + 2}/{max_attempts})."
            )

            time.sleep(delay)
            continue

        documents = [
            Document(
                page_content = match.chunk_text,
                metadata = {
                    "source": match.get("source") or "Unknown",
                    "id": match.id,
                }
            )
            for match in results.matches
        ]

        log_success(
            f"Retrieved {len(documents)} documentation chunks."
        )

        serialized = "\n\n".join(
            f"Source: {doc.metadata['source']}\n"
            f"Content: {doc.page_content}"
            for doc in documents
        )

        return serialized, documents
    return None


def run_llm(query:str)-> Dict[str, Any]:
    """
    Run the RAG pipeline to retrieve relevant documentation
    Returns:
        - answer: The Generated answer
        - context: List of retrieved documentation
    """
    system_prompt = (
        "You are a helpful AI assistant that answers questions about LangChain documentation."
        "You have access to a tool that retrieves relevant documentation. "
        "Use the tool to find relevant information before answering questions."
        "Always cite the sources you use in your answers."
        "If you cannot find the answer in the retrieved documentation, say so."
    )
    agent = create_agent(model, tools=[retrieve_context], system_prompt=system_prompt)
    messages = [{"role":"user", "content": query}]
    response = agent.invoke({"messages":messages})
    answer = response["messages"][-1].content
    context_documents = []
    for message in response["messages"]:
        if isinstance(message, ToolMessage) and hasattr(message, "artifact"):
            if isinstance(message.artifact, list):
                context_documents.append(message.artifact)
    return {
        "answer": answer,
        "context": context_documents
    }
if __name__ == "__main__":
    result = run_llm(query="What are deep agents?")
    print(result)


