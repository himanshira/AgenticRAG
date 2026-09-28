"""Tool instances shared by the agents."""

from crewai_tools import PDFSearchTool, TavilySearchTool

from .config import PDF_PATH

# Chunks, embeds and vector-stores the PDF internally (OpenAI embeddings by default;
# pass config={...} to change the embedder / vector DB).
pdf_tool = PDFSearchTool(pdf=str(PDF_PATH))

# Fresh information from the internet. Needs TAVILY_API_KEY.
web_tool = TavilySearchTool(search_depth="basic", max_results=5)
