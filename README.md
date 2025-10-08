# DocAI

[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](MIT) <!-- Replace with your actual license -->

## Overview

DocAI is a project designed to create a conversational knowledge base from a directory of documents. It automatically watches a specified directory (or share) for new or updated files, uploads their content to a Qdrant vector database, and then leverages a Large Language Model (LLM) provided by Ollama to enable natural language querying of the documents.

## Purpose

The purpose of this project is to serve as a knowledge base you can talk to. It automates the process of indexing documents and making their content accessible through conversational AI. This is useful for:

*   **Knowledge Management:** Easily access information from a collection of documents without manual searching.
*   **Document Understanding:** Get quick summaries and answers to questions about your documents.
*   **Automated Research:** Streamline research by letting the AI synthesize information from multiple sources.

## Application Flow

![application flow](./flow.svg)

## Features

*   **Directory/Share Monitoring:** Automatically detects new or updated documents in a specified directory or network share.
*   **Document Ingestion:**  Supports [PDF, TXT, DOCX, MD] and extracts text content.
*   **Vector Embedding:** Converts document text into vector embeddings.
*   **Qdrant Vector Database Integration:** Stores and indexes vector embeddings in a Qdrant vector database for efficient similarity search.
*   **Ollama LLM Integration:** Leverages Ollama for effortless local hosting of Large Language Models, simplifying setup and ensuring privacy.

## Technologies Used

*   **Python:** The primary programming language.
*   **Qdrant:** Vector database for storing and indexing document embeddings. ([https://qdrant.tech/](https://qdrant.tech/))
*   **Ollama:**  LLM serving and management. ([https://ollama.ai/](https://ollama.ai/))

## License

This project is licensed under the [MIT License](LICENSE) - see the `LICENSE` file for details.

## Future Enhancements

*   Support for more document types.
*   Improve Web interface.
*   Improved document processing and cleaning.