import os
import datetime
from pathlib import Path
import PyPDF2
import subprocess
import ollama
import openai
from jinja2 import Template
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

# Ensure these environment variables are set
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')

openai.api_key = OPENAI_API_KEY

# ChromaDB configuration and functions
class ChromaDBHandler:
    def __init__(self):
        self.CHROMA_PERSIST_DIRECTORY = "./chroma_persist"
        self.openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=OPENAI_API_KEY,
            model_name="text-embedding-ada-002"
        )
        self.chroma_client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=self.CHROMA_PERSIST_DIRECTORY
        ))
        self.collection = self.chroma_client.get_or_create_collection(
            name="research_summaries",
            embedding_function=self.openai_ef
        )

    def add_summary(self, summary, metadata):
        self.collection.add(
            documents=[summary],
            metadatas=[metadata],
            ids=[f"{metadata['title']}_{metadata['date']}"]
        )

    def query_similar(self, query_text, n_results=5):
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results
        )
        return results

    def persist(self):
        self.chroma_client.persist()

# Ollama and OpenAI functions
def ensure_ollama_running():
    try:
        ollama.list()
    except Exception:
        subprocess.run(["ollama", "run", "llama2:7b"], check=True)

def summarize_text_ollama(text):
    response = ollama.generate(model='llama2:7b', prompt=f"Summarize the following abstract in one sentence:\n\n{text}")
    return response['response']

def summarize_text_openai(text):
    response = openai.ChatCompletion.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that summarizes academic research and identifies key themes."},
            {"role": "user", "content": f"Summarize the following abstracts and identify key themes:\n\n{text}"}
        ]
    )
    return response.choices[0].message['content']

# PDF processing functions
def process_new_pdfs(pdf_dir, chroma_handler):
    ensure_ollama_running()
    for pdf_file in Path(pdf_dir).glob('*.pdf'):
        if pdf_file.stat().st_mtime > (datetime.datetime.now() - datetime.timedelta(days=1)).timestamp():
            with open(pdf_file, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                abstract = ""
                for page in reader.pages:
                    text = page.extract_text()
                    if "abstract" in text.lower():
                        abstract = text[text.lower().index("abstract"):].split("\n\n")[0]
                        break
                
                if abstract:
                    summary = summarize_text_ollama(abstract)
                    date = datetime.date.today().isoformat()
                    title = pdf_file.stem
                    
                    # Add to ChromaDB
                    chroma_handler.add_summary(summary, {"title": title, "date": date, "file": pdf_file.name})

def summarize_week(pdf_dir, output_dir, chroma_handler):
    week_ending = datetime.date.today()
    week_start = week_ending - datetime.timedelta(days=7)
    
    abstracts = []
    for pdf_file in Path(pdf_dir).glob('*.pdf'):
        if week_start.timestamp() <= pdf_file.stat().st_mtime <= week_ending.timestamp():
            with open(pdf_file, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    text = page.extract_text()
                    if "abstract" in text.lower():
                        abstract = text[text.lower().index("abstract"):].split("\n\n")[0]
                        abstracts.append(abstract)
                        break
    
    if abstracts:
        combined_abstract = "\n\n".join(abstracts)
        weekly_summary = summarize_text_openai(combined_abstract)
        
        # Query ChromaDB for similar research
        results = chroma_handler.query_similar(weekly_summary)
        
        similar_research = "\n".join([f"- {meta['title']}: {doc}" for meta, doc in zip(results['metadatas'][0], results['documents'][0])])
        
        research_directions = openai.ChatCompletion.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that suggests research problems and directions."},
                {"role": "user", "content": f"Based on this weekly summary and similar research, suggest potential research problems and directions:\n\nWeekly Summary:\n{weekly_summary}\n\nSimilar Research:\n{similar_research}"}
            ]
        ).choices[0].message['content']

        # Generate a joke
        joke = openai.ChatCompletion.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates bad jokes."},
                {"role": "user", "content": "Tell me a bad joke related to scientific research."}
            ]
        ).choices[0].message['content']

        # Render the template
        template = Template("""
# Weekly Research Summary

#### A Summary of awesome recent research in the spaces that exist.

***Thx to sources like TL;DR for their contribution***

{% for theme in themes %}
{{ theme.title }}
{{ theme.summary }}

{% endfor %}

{{ conclusion }}

Similar Research:
{{ similar_research }}

Next Steps: {{ research_directions }}

Joke of the Week: {{ joke }}

:: Nentropic Research 0x0 ::
        """)

        rendered_summary = template.render(
            week_ending=week_ending.isoformat(),
            themes=[{"title": "Theme 1", "summary": "Summary 1"}, {"title": "Theme 2", "summary": "Summary 2"}],  # You'll need to extract these from the OpenAI response
            conclusion="Conclusion goes here",  # Extract this from the OpenAI response
            similar_research=similar_research,
            research_directions=research_directions,
            joke=joke
        )
        
        output_file = Path(output_dir) / f"summary_{week_ending.isoformat()}.md"
        with open(output_file, 'w') as out:
            out.write(rendered_summary)

if __name__ == "__main__":
    pdf_dir = '.'
    weekly_summaries_dir = './weekly_summaries'
    
    chroma_handler = ChromaDBHandler()
    
    process_new_pdfs(pdf_dir, chroma_handler)
    summarize_week(pdf_dir, weekly_summaries_dir, chroma_handler)
    
    # Persist the ChromaDB changes
    chroma_handler.persist()