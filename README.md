# ResearchLens AI

An AI-powered research assistant that searches the web,
generates source-grounded answers, and verifies factual claims.

## Features

- Web research using Tavily
- Gemini-powered answer generation
- Automatic source citations
- Claim extraction
- Claim verification
- Source snippets and URLs
- Gradio web interface
- Automatic Gemini model detection

## Tech Stack

- Python
- Google Gemini
- Tavily
- Gradio

## Architecture

User Query
    ->
Tavily Web Search
    ->
Source Collection
    ->
Gemini
    ->
Cited Answer
    ->
Claim Extraction
    ->
Claim Verification
    ->
Results Dashboard

## Installation

Clone the repository:

git clone YOUR_REPOSITORY_URL

cd research-assistant-with-citations

Install dependencies:

pip install -r requirements.txt

Create a .env file and add:

GEMINI_API_KEY=your_key

TAVILY_API_KEY=your_key

Run:

python app.py

## How It Works

1. User enters a research question.
2. Tavily searches the web.
3. Relevant sources are collected.
4. Gemini generates an answer using only those sources.
5. Factual claims are extracted.
6. Claims are verified against the retrieved sources.
7. The application displays the answer, sources, and verification results.
