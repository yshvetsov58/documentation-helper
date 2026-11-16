import asyncio
import hashlib
import json
import os
import random

import certifi

import ssl
from typing import Any, Dict, List

import pinecone
from dotenv import load_dotenv
from firecrawl.v2.methods.batch import chunk_urls
from langchain_core.documents import Document
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap
from pinecone import Pinecone, SchemaBuilder
from langchain_text_splitters import RecursiveCharacterTextSplitter

from logger import (Colors, log_error, log_header, log_info, log_success, log_warning)
from langchain_openai import OpenAIEmbeddings

load_dotenv()
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
PINECONE_INDEX_NAME = "documentation-openai"
CHUNK_FIELD = "chunk_text"
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small", show_progress_bar=False, chunk_size=50, retry_min_seconds=30
)
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

tavily_extract = TavilyExtract()
tavily_map = TavilyMap()
tavily_craw = TavilyCrawl()
def create_index():
    if not pc.indexes.exists(name=PINECONE_INDEX_NAME):
        schema = (
            SchemaBuilder()
            .add_dense_vector_field(
                "embedding",
                dimension=1536,
                metric="cosine",
            )
            .add_string_field("chunk_text", full_text_search=True)
            .build()
        )
        pc.indexes.create(schema=schema, name=PINECONE_INDEX_NAME)
        log_success(f"Created {PINECONE_INDEX_NAME} index.")
    else:
        log_warning(f"Index {PINECONE_INDEX_NAME} already exists.")

create_index()
index = pc.index(name=PINECONE_INDEX_NAME)

async def main():
    """Main async function to orchestrate ingestion"""
    log_header("Documentation Ingestion pipeline")
    #log_info("""TavilyCrawl: Starting to Crawl documentation from https://docs.langchain.com/oss/python/langchain/overview""",
    #         Colors.PURPLE)
      #  res = tavily_craw.invoke({
      #      "url" : "https://docs.langchain.com/oss/python/langchain/overview",
      #      "max_depth" : 2,
      #      "extract_depth" : "advanced",
      #      "instructions":"LangChain documentation",
      #  })
      # all_docs = [Document(page_content= result['raw_content'], metadata = {"source": result["url"]}) for result in res['results']]
      #  log_success(f"TavilyCrawl: Successfully crawled {len(all_docs)} documents.")
    site_map = await tavily_map.ainvoke("https://docs.langchain.com/oss/python/langchain/overview")
    log_success(f"TavilyMap: Successfully mapped {len(site_map["results"])} URLs from documentation site.")
    url_batches = chunk_urls(list(site_map["results"]), chunk_size=20)
    log_info(f"URL Processing Split {len(site_map['results'])} URLs into {len(url_batches)} batches.",
             Colors.BLUE)
    all_documents = await async_extract(url_batches)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(iter(all_documents))
    texts = [doc.page_content for doc in split_docs]
    vectors = await embeddings.aembed_documents(texts)
    prepared_docs = [
        {"doc": doc, "embedding": vector}
        for doc, vector in zip(split_docs, vectors, strict=True)
    ]
    log_success(f"Text Splitter: Created {len(split_docs)} chunks from {len(all_documents)} documents.")
    await async_save_docs_into_index(prepared_docs, batch_size = 500)

async def extract_batch(urls: List[str], batch_num: int) -> List[Dict[str, Any]]:
    """Extract documents from a batch of URLs"""
    try:
        log_info(f"Extracting {len(urls)} documents from {batch_num} batches.", Colors.BLUE)
        docs = await tavily_extract.ainvoke(input={"urls": urls})
        log_success(f"Extracted {len(docs)} documents from {batch_num} batches.")
        return docs
    except Exception as e:
        log_error(f"TavilyExtract Error: {e}")
        return []

async def async_extract(url_batches:List[List[str]]):
    log_header("Document Extraction pipeline")
    log_info(f"TavilyExtract: Starting concurrent extraction of {len(url_batches)} batches.", Colors.DARKCYAN)
    tasks = [extract_batch(batch, i+1) for i, batch in enumerate(url_batches)]
    results = await asyncio.gather(*tasks)
    all_pages = []
    failed_batches = 0
    for res in results:
        if isinstance(res, Exception):
            log_error(f"TavilyExtract Error: {res}")
            failed_batches += 1
        else:
            for extracted_page in res["results"]:
                document = Document(
                    page_content=extracted_page["raw_content"],
                    metadata={"source": extracted_page["url"]},
                )
                all_pages.append(document)
    return all_pages

async def add_batch(docs: List[Document], batch_num: int):
    log_header("Insert documents into vector store.")
    log_info(f"Inserting {len(docs)} documents into vector store.")
    documents = [
        {
            **doc.metadata,
            "_id":make_document_id(doc),
            CHUNK_FIELD: doc.page_content,
        }
        for i, doc in enumerate(docs)
    ]
    try:
        result = await asyncio.to_thread(
            index.documents.batch_upsert,
            namespace="langchain-docs-v2",
            documents=documents,
        )

        if result.errors:
            for error in result.errors:
                log_error(
                    f"Batch {batch_num}: {error.error_message}"
                )
            return False

        return True
    except Exception as error:
        log_error(f"Batch {batch_num}: {error}")
        return False

async def async_save_docs_into_index(docs: list[dict[str, Any]], batch_size: int = 50):
    unique_documents = {}
    for item in docs:
        doc = item["doc"]
        vector = item["embedding"]

        doc_id = make_document_id(doc)
        unique_documents[doc_id] = {
            **doc.metadata,
            "_id": doc_id,
            "chunk_text": doc.page_content,
            "embedding":vector
        }
    pending = list(unique_documents.values())
    total = len(pending)

    if not pending:
        log_info("No documents to upload.")
        return

    max_attempts = 8

    for attempt in range(max_attempts):
        log_info(
            f"Attempt {attempt + 1}/{max_attempts}: "
            f"{len(pending)} documents remaining."
        )

        result = await asyncio.to_thread(
            index.documents.batch_upsert,
            namespace="langchain-docs-v2",
            documents=pending,
            batch_size=min(batch_size, 50),
            max_concurrency=1,
            timeout=120,
        )

        permanent_errors = [
            error for error in result.errors
            if not error.retryable
        ]

        if permanent_errors:
            raise RuntimeError(
                "Non-retryable upload errors:\n"
                + "\n".join(
                    error.error_message
                    for error in permanent_errors
                )
            )

        pending = list(result.failed_items)

        if not pending:
            log_success(f"All {total} documents accepted by Pinecone.")
            return

        if attempt == max_attempts - 1:
            break

        delay = min(60 * (2 ** attempt), 180) + random.uniform(0, 5)

        log_warning(
            f"{len(pending)} documents failed. "
            f"Retrying in {delay:.0f} seconds."
        )

        await asyncio.sleep(delay)

    raise RuntimeError(
        f"Upload incomplete after {max_attempts} attempts: "
        f"{len(pending)} documents remaining."
    )

def make_document_id(doc: Document) -> str:
    identity = json.dumps(
        [doc.metadata["source"], doc.page_content],
        ensure_ascii=False,
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()

if __name__ == "__main__":
    asyncio.run(main())