import io
import os
import re
from typing import Dict, List, Optional

import google.generativeai as genai
import requests
import streamlit as st
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from pypdf import PdfReader

load_dotenv()

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="ScholarGPT Pro", page_icon="📚", layout="wide")

# =========================
# CUSTOM CSS
# =========================
st.markdown(
    """
<style>
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}
.main-title {
    font-size: 2.4rem;
    font-weight: 800;
    margin-bottom: 0.2rem;
}
.sub-title {
    color: #9aa4b2;
    margin-bottom: 1.2rem;
}
.info-card {
    background: linear-gradient(135deg, #0f2740, #112d4e);
    border-radius: 16px;
    padding: 16px 18px;
    margin-bottom: 12px;
    border: 1px solid rgba(255,255,255,0.08);
}
.paper-card {
    background: #111827;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 16px;
    margin-bottom: 12px;
}
.small-muted {
    color: #9aa4b2;
    font-size: 0.92rem;
}
hr {
    border-color: rgba(255,255,255,0.08) !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# =========================
# CONSTANTS
# =========================
APP_TITLE = "📚 ScholarGPT Pro"
APP_SUBTITLE = "Upload papers, search many papers, add links, and chat with disciplined citations."
DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_MIN_YEAR = 2018
DEFAULT_MAX_YEAR = 2025
TARGET_PARAGRAPH_LINES = 9
MIN_PARAGRAPH_LINES = 6
MAX_PARAGRAPH_LINES = 9
MAX_CONTEXT_CHARS = 140000

DEFAULT_SYSTEM_PROMPT = """
You are a research assistant specializing in academic writing with APA 7th edition formatting.

RESPONSE FORMAT:
1. Write in clear academic prose as a narrative essay
2. Use proper APA 7th edition in-text citations (Author, Year) naturally
3. Provide a "References" section at the end with proper APA 7th edition formatting
4. NO subheadings, NO bullet points - write as continuous narrative

CITATION RULES:
- ONLY use citations from uploaded documents and searched papers provided below
- Use (Author, Year) format for in-text citations
- For multiple authors: (Smith & Johnson, 2023) for 2 authors, (Smith et al., 2023) for 3+ authors
- If no author: (Organization, Year) or ("Title of Work", Year)
- Place citations naturally where they make sense, not forced
- Add 1-2 citations in introduction if relevant
- NEVER invent citations or use sources not provided

WRITING STYLE:
- Write as continuous narrative paragraphs
- Use complete words: write "bullying" instead of "bully", "trustworthiness" instead of "truacy"
- NO abbreviations like "vat" - write "value-added tax" or full terms
- NO subheadings like "Introduction" - just write the content
- NO bullet points or numbered lists
- Professional academic tone throughout

CONTENT RULES:
1. Answer from the supplied paper context first
2. Use uploaded documents for detailed content (full text available)
3. Use searched papers for metadata-based answers (title, authors, journal, year)
4. If information isn't in the provided sources, say so clearly
5. Be clear, accurate, and academic in tone
6. Structure answers logically with proper flow

REFERENCES SECTION:
- List all cited sources in proper APA 7th edition format
- For uploaded documents: Use filename as title if no clear author
- For searched papers: Use available metadata (authors, year, title, journal)
- Format: Author, A. A. (Year). Title of work. Journal Name, volume(issue), pages. OR DOI
- Only include sources that were actually cited in the text
"""

USER_AGENT = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

# =========================
# SESSION STATE
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "documents" not in st.session_state:
    st.session_state.documents = []
if "paper_search_results" not in st.session_state:
    st.session_state.paper_search_results = []
if "combined_context" not in st.session_state:
    st.session_state.combined_context = ""
if "last_answer" not in st.session_state:
    st.session_state.last_answer = ""
if "saved_histories" not in st.session_state:
    st.session_state.saved_histories = []

# =========================
# HISTORY MANAGEMENT
# =========================
import json
from datetime import datetime

def save_chat_history(history_name: str) -> bool:
    """Save current chat history, documents, and search results"""
    try:
        history_data = {
            "name": history_name,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "messages": st.session_state.messages,
            "documents": st.session_state.documents,
            "paper_search_results": st.session_state.paper_search_results,
            "combined_context": st.session_state.combined_context
        }
        
        # Save to file
        filename = f"chat_history_{history_name.replace(' ', '_').lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join("histories", filename)
        
        # Create histories directory if it doesn't exist
        os.makedirs("histories", exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(history_data, f, ensure_ascii=False, indent=2)
        
        # Add to session state
        st.session_state.saved_histories.append({
            "name": history_name,
            "filename": filename,
            "timestamp": history_data["timestamp"],
            "filepath": filepath
        })
        
        return True
    except Exception as e:
        st.error(f"Failed to save history: {e}")
        return False

def load_chat_history(filepath: str) -> bool:
    """Load a saved chat history"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            history_data = json.load(f)
        
        # Restore session state
        st.session_state.messages = history_data.get("messages", [])
        st.session_state.documents = history_data.get("documents", [])
        st.session_state.paper_search_results = history_data.get("paper_search_results", [])
        st.session_state.combined_context = history_data.get("combined_context", "")
        
        return True
    except Exception as e:
        st.error(f"Failed to load history: {e}")
        return False

def get_saved_histories() -> List[Dict]:
    """Get list of all saved histories"""
    histories = []
    try:
        if os.path.exists("histories"):
            for filename in os.listdir("histories"):
                if filename.endswith(".json"):
                    filepath = os.path.join("histories", filename)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        histories.append({
                            "name": data.get("name", filename),
                            "filename": filename,
                            "timestamp": data.get("timestamp", ""),
                            "filepath": filepath
                        })
                    except:
                        continue
    except Exception as e:
        st.error(f"Error loading histories: {e}")
    
    return sorted(histories, key=lambda x: x["timestamp"], reverse=True)

def delete_chat_history(filepath: str) -> bool:
    """Delete a saved chat history"""
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
            # Remove from session state
            st.session_state.saved_histories = [
                h for h in st.session_state.saved_histories 
                if h["filepath"] != filepath
            ]
            return True
    except Exception as e:
        st.error(f"Failed to delete history: {e}")
    return False

# =========================
# HELPERS
# =========================
def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_year(text: str) -> str:
    if not text:
        return ""
    years = re.findall(r"(?:19|20)\d{2}", text[:5000])
    valid = [y for y in years if DEFAULT_MIN_YEAR <= int(y) <= DEFAULT_MAX_YEAR]
    return valid[0] if valid else ""


def extract_authors_from_text(text: str) -> List[str]:
    if not text:
        return []
    head = text[:3000].replace("\n", " ")
    patterns = [
        r"By\s+([A-Z][A-Za-z'`.-]+(?:\s+[A-Z][A-Za-z'`.-]+){0,3})",
        r"Authors?\s*[:\-]\s*([A-Z][A-Za-z'`.-]+(?:\s+[A-Z][A-Za-z'`.-]+){0,3})",
    ]
    found: List[str] = []
    for pattern in patterns:
        for match in re.findall(pattern, head):
            candidate = clean_text(match)
            if candidate and len(candidate.split()) <= 4 and candidate not in found:
                found.append(candidate)
    return found[:3]


def surname_from_name(name: str) -> str:
    parts = [p for p in re.split(r"\s+", clean_text(name)) if p]
    if not parts:
        return "Source"
    return parts[-1].strip(",.;:)")


def build_citation_label(title: str = "", authors: Optional[List[str]] = None, year: str = "") -> str:
    authors = authors or []
    if authors:
        surname = surname_from_name(authors[0])
    else:
        surname = surname_from_name(title or "Source")
    return f"({surname}, {year or 'n.d.'})"


def enrich_document_metadata(
    name: str,
    source: str,
    text: str,
    title: Optional[str] = None,
    authors: Optional[List[str]] = None,
    year: str = "",
) -> Dict:
    authors = authors or extract_authors_from_text(text)
    year = year or extract_year(text)
    title = title or name
    citation_label = build_citation_label(title=title, authors=authors, year=year)
    return {
        "name": name,
        "source": source,
        "title": title,
        "text": text,
        "authors": authors,
        "year": year,
        "citation_label": citation_label,
    }


def extract_pdf_text_from_file(uploaded_file) -> str:
    try:
        pdf = PdfReader(uploaded_file)
        pages = [(page.extract_text() or "") for page in pdf.pages]
        return clean_text("\n".join(pages))
    except Exception as e:
        return f"ERROR_READING_PDF_FILE: {e}"


def extract_pdf_text_from_url(url: str) -> Dict:
    try:
        response = requests.get(url, headers=USER_AGENT, timeout=40)
        response.raise_for_status()
        file_like = io.BytesIO(response.content)
        pdf = PdfReader(file_like)
        pages = [(page.extract_text() or "") for page in pdf.pages]
        text = clean_text("\n".join(pages))
        title = url.split("/")[-1] or "paper_link.pdf"
        return {
            "text": text,
            "title": title,
            "authors": extract_authors_from_text(text),
            "year": extract_year(text),
        }
    except Exception as e:
        return {"text": f"ERROR_READING_PDF_URL: {e}", "title": "", "authors": [], "year": ""}


def extract_html_text_from_url(url: str) -> Dict:
    try:
        response = requests.get(url, headers=USER_AGENT, timeout=40)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
            tag.decompose()

        title = soup.title.string.strip() if soup.title and soup.title.string else ""
        description = ""
        abstract = ""
        authors: List[str] = []
        year = ""

        desc = soup.find("meta", attrs={"name": "description"})
        if desc and desc.get("content"):
            description = desc["content"].strip()

        abstract_tag = soup.find("meta", attrs={"name": "citation_abstract"})
        if abstract_tag and abstract_tag.get("content"):
            abstract = abstract_tag["content"].strip()

        for author_tag in soup.find_all("meta", attrs={"name": "citation_author"}):
            if author_tag.get("content"):
                candidate = clean_text(author_tag["content"])
                if candidate and candidate not in authors:
                    authors.append(candidate)

        year_tag = soup.find("meta", attrs={"name": "citation_publication_date"}) or soup.find(
            "meta", attrs={"name": "citation_date"}
        )
        if year_tag and year_tag.get("content"):
            match = re.search(r"(?:19|20)\d{2}", year_tag["content"])
            if match:
                year = match.group(0)

        body_text = clean_text(soup.get_text(separator=" ", strip=True))
        text = clean_text(f"Title: {title}\nDescription: {description}\nAbstract: {abstract}\nBody: {body_text}")
        return {"text": text, "title": title or url.split("/")[-1] or "paper_link", "authors": authors, "year": year}
    except Exception as e:
        return {"text": f"ERROR_READING_HTML_URL: {e}", "title": "", "authors": [], "year": ""}


def extract_text_from_url(url: str) -> Dict:
    url = url.strip()
    if url.lower().endswith(".pdf"):
        return extract_pdf_text_from_url(url)
    return extract_html_text_from_url(url)


def build_combined_context(documents: List[Dict], search_results: List[Dict], max_chars: int = MAX_CONTEXT_CHARS) -> str:
    if not documents and not search_results:
        return ""

    parts: List[str] = []
    total = 0

    for doc in documents:
        block = (
            f"\n\n===== SOURCE TYPE: DOCUMENT OR LINK =====\n"
            f"TITLE: {doc.get('title') or doc.get('name')}\n"
            f"AUTHORS: {', '.join(doc.get('authors') or []) or 'Unknown'}\n"
            f"YEAR: {doc.get('year') or 'Unknown'}\n"
            f"CITATION LABEL: {doc.get('citation_label') or ''}\n"
            f"SOURCE: {doc.get('source')}\n"
            f"FULL TEXT: {doc.get('text', '')}\n"
        )
        if total + len(block) > max_chars:
            remaining = max_chars - total
            if remaining > 0:
                parts.append(block[:remaining])
            break
        parts.append(block)
        total += len(block)

    for paper in search_results:
        block = (
            f"\n\n===== SOURCE TYPE: SEARCHED PAPER =====\n"
            f"TITLE: {paper['title']}\n"
            f"AUTHORS: {paper['authors']}\n"
            f"YEAR: {paper['year']}\n"
            f"JOURNAL: {paper['journal']}\n"
            f"DOI: {paper['doi']}\n"
            f"LINK: {paper['link']}\n"
            f"CITATION LABEL: {paper.get('citation_label', '')}\n"
            f"ABSTRACT: {paper.get('abstract', '')}\n"
            f"NOTE: Use only metadata and abstract for this searched paper.\n"
        )
        if total + len(block) > max_chars:
            remaining = max_chars - total
            if remaining > 0:
                parts.append(block[:remaining])
            break
        parts.append(block)
        total += len(block)

    return "".join(parts).strip()


def get_gemini_api_key() -> str:
    try:
        key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        key = ""
    if not key:
        key = os.getenv("GEMINI_API_KEY", "")
    return key.strip()


def configure_gemini(api_key: str):
    genai.configure(api_key=api_key)


def search_crossref_papers(query: str, rows: int = 50, min_year: int = DEFAULT_MIN_YEAR, max_year: int = DEFAULT_MAX_YEAR) -> List[Dict]:
    try:
        url = "https://api.crossref.org/works"
        
        # Check if query mentions Nigeria and filter for Nigeria scholars
        nigeria_keywords = ["nigeria", "nigerian", "africa", "african"]
        is_nigeria_search = any(keyword.lower() in query.lower() for keyword in nigeria_keywords)
        
        params = {
            "query": query,
            "rows": rows * 3 if is_nigeria_search else rows * 2,  # Get more results for Nigeria searches to filter
            "select": "title,URL,DOI,author,issued,container-title,abstract",
        }
        
        # Add Nigeria-specific filtering if searching for Nigeria content
        if is_nigeria_search:
            # Try to find Nigeria-related papers
            nigeria_query = f"{query} Nigeria"
            params["query"] = nigeria_query
            
        response = requests.get(url, params=params, headers=USER_AGENT, timeout=40)
        response.raise_for_status()
        data = response.json()

        results: List[Dict] = []
        for item in data.get("message", {}).get("items", []):
            title = item.get("title", ["Untitled"])[0]
            journal = item.get("container-title", [""])[0]
            doi = item.get("DOI", "")
            link = item.get("URL", "")
            issued = item.get("issued", {})
            date_parts = issued.get("date-parts", [])
            year = ""
            if date_parts and date_parts[0]:
                year = str(date_parts[0][0])
            if not year:
                continue
            try:
                year_int = int(year)
            except ValueError:
                continue
            if not (min_year <= year_int <= max_year):
                continue

            authors: List[str] = []
            nigerian_authors = 0
            for author in item.get("author", []):
                given = author.get("given", "")
                family = author.get("family", "")
                name = clean_text(f"{given} {family}")
                if name:
                    # Check for Nigerian indicators in author names or affiliations
                    if any(indicator.lower() in name.lower() for indicator in ["ade", "ayo", "oba", "oke", "nig", "lagos", "abuja", "kano", "ibadan"]):
                        nigerian_authors += 1
                    authors.append(name)

            # For Nigeria searches, prioritize papers with Nigerian authors or Nigeria-related content
            if is_nigeria_search:
                # Check if title mentions Nigeria or if there are Nigerian authors
                title_nigeria = any(keyword.lower() in title.lower() for keyword in nigeria_keywords)
                abstract = clean_text(re.sub(r"<[^>]+>", " ", item.get("abstract", "") or ""))
                abstract_nigeria = any(keyword.lower() in abstract.lower() for keyword in nigeria_keywords)
                
                # Include if Nigeria-related in title/abstract OR has Nigerian authors
                if not (title_nigeria or abstract_nigeria or nigerian_authors > 0):
                    continue

            abstract = clean_text(re.sub(r"<[^>]+>", " ", item.get("abstract", "") or ""))
            source = {
                "title": title,
                "journal": journal,
                "year": year,
                "doi": doi,
                "link": link,
                "authors": ", ".join(authors[:8]) if authors else "N/A",
                "abstract": abstract,
                "citation_label": build_citation_label(title=title, authors=authors, year=year),
            }
            results.append(source)
            if len(results) >= rows:
                break

        return results
    except Exception as e:
        st.error(f"Paper search failed: {e}")
        return []


def build_source_list(documents: List[Dict], search_results: List[Dict]) -> str:
    if not documents and not search_results:
        return "No loaded sources."
    
    lines = ["\n=== AVAILABLE SOURCES FOR CITATIONS ==="]
    
    # Uploaded documents
    if documents:
        lines.append("\nUPLOADED DOCUMENTS (Full text available):")
        for i, doc in enumerate(documents, start=1):
            authors = ', '.join(doc.get('authors', [])[:3]) or "Unknown author"
            if len(doc.get('authors', [])) > 3:
                authors += " et al."
            year = doc.get('year', 'n.d.')
            title = doc.get('title') or doc.get('name', 'Untitled')
            source = doc.get('source', 'Unknown source')
            lines.append(f"[{i}] {authors} ({year}). {title}. [Uploaded document from: {source}]")
    
    # Searched papers
    if search_results:
        lines.append("\nSEARCHED PAPERS (Metadata and abstract available):")
        for i, paper in enumerate(search_results, start=1):
            authors = paper.get('authors', 'N/A')
            year = paper.get('year', 'n.d.')
            title = paper.get('title', 'Untitled')
            journal = paper.get('journal', 'N/A')
            doi = paper.get('doi', '')
            lines.append(f"[{i}] {authors} ({year}). {title}. {journal}. DOI: {doi}")
    
    lines.append("\n=== END OF SOURCES ===")
    return "\n".join(lines)


def chunk_sentences_into_lines(text: str) -> List[str]:
    sentences = [clean_text(s) for s in re.split(r"(?<=[.!?])\s+", text) if clean_text(s)]
    if not sentences:
        return []
    if len(sentences) >= TARGET_PARAGRAPH_LINES:
        return sentences[:TARGET_PARAGRAPH_LINES]

    lines: List[str] = []
    current = ""
    for sent in sentences:
        candidate = f"{current} {sent}".strip()
        if current and len(candidate.split()) > 18:
            lines.append(current.strip())
            current = sent
        else:
            current = candidate
    if current:
        lines.append(current.strip())

    if len(lines) == 1:
        words = lines[0].split()
        chunk = max(6, len(words) // TARGET_PARAGRAPH_LINES or 6)
        lines = [" ".join(words[i : i + chunk]) for i in range(0, len(words), chunk)]

    while len(lines) < MIN_PARAGRAPH_LINES:
        lines.append("This source-based explanation further clarifies the discussion.")

    if len(lines) > MAX_PARAGRAPH_LINES:
        merged: List[str] = []
        step = max(1, round(len(lines) / MAX_PARAGRAPH_LINES))
        for i in range(0, len(lines), step):
            merged.append(" ".join(lines[i : i + step]).strip())
        lines = merged[:MAX_PARAGRAPH_LINES]

    return lines[:MAX_PARAGRAPH_LINES]


def choose_three_citations(documents: List[Dict], search_results: List[Dict]) -> List[str]:
    labels: List[str] = []
    for source in documents + search_results:
        label = source.get("citation_label")
        if label and label not in labels:
            labels.append(label)
    if not labels:
        return []
    if len(labels) == 1:
        return [labels[0], labels[0], labels[0]]
    if len(labels) == 2:
        return [labels[0], labels[1], labels[0]]
    return labels[:3]


def enforce_paragraph_citations(answer: str, documents: List[Dict], search_results: List[Dict]) -> str:
    citations = choose_three_citations(documents, search_results)
    if not answer.strip() or not citations:
        return answer

    blocks = [b.strip() for b in re.split(r"\n\s*\n", answer) if b.strip()]
    processed: List[str] = []

    for block in blocks:
        if block.startswith("#"):
            processed.append(block)
            continue

        plain = re.sub(r"\([A-Za-z][^\)]*?,\s*(?:19|20)\d{2}|n\.d\.\)", "", block)
        plain = clean_text(plain)
        lines = chunk_sentences_into_lines(plain)
        if not lines:
            continue

        top_idx = 2 if len(lines) >= 3 else 0
        mid_idx = 4 if len(lines) >= 6 else min(len(lines) - 1, len(lines) // 2)
        bottom_idx = max(0, len(lines) - 2)

        lines[top_idx] = f"{lines[top_idx]} {citations[0]}".strip()
        if mid_idx != top_idx:
            lines[mid_idx] = f"{lines[mid_idx]} {citations[1]}".strip()
        else:
            alt_idx = min(len(lines) - 1, top_idx + 1)
            lines[alt_idx] = f"{lines[alt_idx]} {citations[1]}".strip()
        if bottom_idx not in {top_idx, mid_idx}:
            lines[bottom_idx] = f"{lines[bottom_idx]} {citations[2]}".strip()
        else:
            lines[-1] = f"{lines[-1]} {citations[2]}".strip()

        processed.append("\n".join(lines))

    return "\n\n".join(processed)


def generate_answer(
    api_key: str,
    model_name: str,
    system_prompt: str,
    context_text: str,
    chat_history: List[Dict],
    user_query: str,
    documents: List[Dict],
    search_results: List[Dict],
    add_three_citations: bool = True,
) -> str:
    configure_gemini(api_key)
    model = genai.GenerativeModel(model_name)

    recent_history = chat_history[-10:] if chat_history else []
    history_text = ""
    for msg in recent_history:
        history_text += f"{msg.get('role', 'user').upper()}: {msg.get('content', '')}\n"

    source_list = build_source_list(documents, search_results)

    prompt = f"""
{system_prompt}

AVAILABLE SOURCES:
{source_list}

DOCUMENT CONTEXT:
{context_text if context_text else 'No document context is loaded yet.'}

CHAT HISTORY:
{history_text if history_text else 'No previous chat history.'}

USER QUESTION:
{user_query}

IMPORTANT INSTRUCTIONS:
1. Use ONLY the sources listed in AVAILABLE SOURCES above
2. For uploaded documents: you have full text content available
3. For searched papers: you have metadata (title, authors, year, journal) and possibly abstract
4. Include proper APA 7th edition in-text citations: (Author, Year) naturally
5. End with a "References" section listing all cited sources in APA 7th edition format
6. Format references as:
   - For uploaded docs: Author, A. A. (Year). Title of document. Retrieved from [source]
   - For searched papers: Author, A. A. (Year). Title of article. Journal Name, volume(issue), pages. https://doi.org/xxxx
7. NEVER invent citations or use sources not provided
8. If no information is available, say so clearly
9. Write as continuous narrative - NO subheadings, NO bullet points
10. Use complete words: "bullying" not "bully", "trustworthiness" not "truacy", "value-added tax" not "vat"
11. Place citations naturally where appropriate, not forced
"""

    response = model.generate_content(prompt)
    answer = (response.text or "").strip() or "No response was returned."
    if add_three_citations:
        answer = enforce_paragraph_citations(answer, documents, search_results)
    return answer


def summarize_document(api_key: str, model_name: str, document_name: str, document_text: str) -> str:
    configure_gemini(api_key)
    model = genai.GenerativeModel(model_name)
    prompt = f"""
Summarize this document under these headings:
1. Title/Source
2. Main Topic
3. Objectives
4. Method
5. Key Findings
6. Conclusion

Document name: {document_name}
Document text:
{document_text[:35000]}
"""
    response = model.generate_content(prompt)
    return (response.text or "").strip() or "No summary returned."

# =========================
# SIDEBAR
# =========================
st.sidebar.title("⚙️ Settings")
api_key = get_gemini_api_key()
manual_api_key = st.sidebar.text_input("Gemini API Key", type="password", value="" if api_key else "")
if manual_api_key.strip():
    api_key = manual_api_key.strip()

model_name = st.sidebar.selectbox(
    "Gemini Model",
    options=["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.5-pro"],
    index=0,
)

system_prompt = st.sidebar.text_area("System Prompt", value=DEFAULT_SYSTEM_PROMPT, height=220)
add_three_citations = st.sidebar.checkbox("Enforce 3 citations per paragraph", value=False)
search_rows = st.sidebar.slider("Number of papers to fetch", min_value=10, max_value=100, value=50, step=10)
st.sidebar.caption(
    "Citation mode: one citation near the top line, one around the middle line, and one near the bottom line of each paragraph."
)

if st.sidebar.button("🧹 Clear Chat History"):
    st.session_state.messages = []
    st.session_state.last_answer = ""
    st.rerun()

if st.sidebar.button("🗑️ Clear All Documents"):
    st.session_state.documents = []
    st.session_state.paper_search_results = []
    st.session_state.combined_context = ""
    st.rerun()

# =========================
# HISTORY MANAGEMENT
# =========================
st.sidebar.markdown("---")
st.sidebar.subheader("📚 Chat History")

# Save current history
history_name = st.sidebar.text_input("History name", placeholder="e.g., Research on Education")
if st.sidebar.button("💾 Save Current Chat"):
    if not history_name.strip():
        st.sidebar.warning("Please enter a history name.")
    elif not st.session_state.messages:
        st.sidebar.warning("No messages to save.")
    else:
        if save_chat_history(history_name.strip()):
            st.sidebar.success(f"History '{history_name}' saved successfully!")
        else:
            st.sidebar.error("Failed to save history.")

# Load saved histories
st.sidebar.markdown("**Saved Histories:**")
saved_histories = get_saved_histories()

if saved_histories:
    for history in saved_histories[:5]:  # Show last 5 histories
        col1, col2, col3 = st.sidebar.columns([3, 1, 1])
        with col1:
            st.sidebar.write(f"📄 {history['name']}")
            st.sidebar.caption(f"📅 {history['timestamp']}")
        with col2:
            if st.sidebar.button("📂", key=f"load_{history['filepath']}"):
                if load_chat_history(history['filepath']):
                    st.sidebar.success(f"Loaded '{history['name']}'")
                    st.rerun()
        with col3:
            if st.sidebar.button("🗑️", key=f"del_{history['filepath']}"):
                if delete_chat_history(history['filepath']):
                    st.sidebar.success(f"Deleted '{history['name']}'")
                    st.rerun()
else:
    st.sidebar.caption("No saved histories yet.")

# =========================
# HEADER
# =========================
st.markdown(f'<div class="main-title">{APP_TITLE}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sub-title">{APP_SUBTITLE}</div>', unsafe_allow_html=True)

if not api_key:
    st.warning("Add your GEMINI_API_KEY in Streamlit secrets or paste it in the sidebar.")

st.info(
    "This version filters searched papers to 2018-2025 and formats paragraphs to carry exactly three separate citations where sources are available."
)

tab1, tab2, tab3 = st.tabs(["📄 Upload Papers", "🔗 Add Paper Link / Search Papers", "💬 Chat With Papers"])

# =========================
# TAB 1
# =========================
with tab1:
    st.subheader("Upload PDF Papers")
    uploaded_files = st.file_uploader("Upload one or more PDF files", type=["pdf"], accept_multiple_files=True)

    if uploaded_files and st.button("Process Uploaded PDFs"):
        new_docs: List[Dict] = []
        with st.spinner("Reading uploaded PDFs..."):
            for file in uploaded_files:
                text = extract_pdf_text_from_file(file)
                if text and not text.startswith("ERROR_READING_PDF_FILE"):
                    new_docs.append(
                        enrich_document_metadata(name=file.name, source="uploaded_pdf", text=text, title=file.name)
                    )
                else:
                    st.error(f"Could not read {file.name}: {text}")
        if new_docs:
            st.session_state.documents.extend(new_docs)
            st.session_state.combined_context = build_combined_context(
                st.session_state.documents, st.session_state.paper_search_results
            )
            st.success(f"Processed {len(new_docs)} uploaded PDF(s).")

    if st.session_state.documents:
        st.markdown("### Loaded Documents")
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown(
                f'<div class="info-card"><b>Loaded documents:</b> {len(st.session_state.documents)}</div>',
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                f'<div class="info-card"><b>Searched papers:</b> {len(st.session_state.paper_search_results)}</div>',
                unsafe_allow_html=True,
            )

        for i, doc in enumerate(st.session_state.documents, start=1):
            with st.expander(f"{i}. {doc['title']}"):
                st.markdown(f"**Citation label:** {doc.get('citation_label', 'N/A')}")
                st.markdown(f"**Detected year:** {doc.get('year') or 'Unknown'}")
                st.markdown(f"**Detected authors:** {', '.join(doc.get('authors') or []) or 'Unknown'}")
                preview = doc["text"][:2500] + ("..." if len(doc["text"]) > 2500 else "")
                st.text_area("Preview", preview, height=220, key=f"preview_{i}")

# =========================
# TAB 2
# =========================
with tab2:
    st.subheader("Add a Paper Link")
    paper_url = st.text_input("Paste a paper URL or direct PDF link")

    if st.button("Read This Link"):
        if not paper_url.strip():
            st.warning("Paste a valid URL first.")
        else:
            with st.spinner("Reading the link..."):
                extracted = extract_text_from_url(paper_url)
                if extracted.get("text", "").startswith("ERROR_READING_"):
                    st.error(extracted["text"])
                else:
                    name_guess = extracted.get("title") or paper_url.split("/")[-1] or "paper_link"
                    st.session_state.documents.append(
                        enrich_document_metadata(
                            name=name_guess,
                            source=paper_url,
                            text=extracted.get("text", ""),
                            title=extracted.get("title") or name_guess,
                            authors=extracted.get("authors", []),
                            year=extracted.get("year", ""),
                        )
                    )
                    st.session_state.combined_context = build_combined_context(
                        st.session_state.documents, st.session_state.paper_search_results
                    )
                    st.success("Link content added successfully.")

    st.markdown("---")
    st.subheader("Search Papers")
    search_query = st.text_input("Search by title, topic, or keywords")

    if st.button("Search Papers"):
        if not search_query.strip():
            st.warning("Enter a search query.")
        else:
            # Check if this is a Nigeria-focused search
            nigeria_keywords = ["nigeria", "nigerian", "africa", "african"]
            is_nigeria_search = any(keyword.lower() in search_query.lower() for keyword in nigeria_keywords)
            
            if is_nigeria_search:
                st.info("🇳🇬 **Nigeria Scholar Filter Activated**: Showing papers by Nigerian scholars and Nigeria-related research")
            
            with st.spinner("Searching papers..."):
                results = search_crossref_papers(
                    search_query, rows=search_rows, min_year=DEFAULT_MIN_YEAR, max_year=DEFAULT_MAX_YEAR
                )
                st.session_state.paper_search_results = results
                st.session_state.combined_context = build_combined_context(
                    st.session_state.documents, st.session_state.paper_search_results
                )

    if st.session_state.paper_search_results:
        # Check if last search was Nigeria-focused
        nigeria_keywords = ["nigeria", "nigerian", "africa", "african"]
        is_nigeria_search = any(keyword.lower() in search_query.lower() for keyword in nigeria_keywords) if 'search_query' in locals() else False
        
        if is_nigeria_search:
            st.markdown(f"### 🇳🇬 Nigeria-Focused Search Results ({len(st.session_state.paper_search_results)})")
        else:
            st.markdown(f"### Search Results ({len(st.session_state.paper_search_results)})")
            
        for idx, item in enumerate(st.session_state.paper_search_results, start=1):
            st.markdown('<div class="paper-card">', unsafe_allow_html=True)
            st.markdown(f"**{idx}. {item['title']}**")
            st.markdown(f"<div class='small-muted'><b>Citation label:</b> {item['citation_label']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='small-muted'><b>Authors:</b> {item['authors']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='small-muted'><b>Journal:</b> {item['journal'] or 'N/A'}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='small-muted'><b>Year:</b> {item['year'] or 'N/A'}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='small-muted'><b>DOI:</b> {item['doi'] or 'N/A'}</div>", unsafe_allow_html=True)
            if item["abstract"]:
                st.markdown(f"<div class='small-muted'><b>Abstract preview:</b> {item['abstract'][:350]}...</div>", unsafe_allow_html=True)
            if item["link"]:
                st.markdown(f"[Open Paper Link]({item['link']})")
            st.markdown("</div>", unsafe_allow_html=True)

# =========================
# TAB 3
# =========================
with tab3:
    st.subheader("Chat With Your Papers")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown(
            f'<div class="info-card"><b>Uploaded documents:</b> {len(st.session_state.documents)}</div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f'<div class="info-card"><b>Searched papers:</b> {len(st.session_state.paper_search_results)}</div>',
            unsafe_allow_html=True,
        )

    if st.session_state.documents and api_key and st.button("Summarize Last Added Document"):
        last_doc = st.session_state.documents[-1]
        with st.spinner("Summarizing document..."):
            summary = summarize_document(
                api_key=api_key,
                model_name=model_name,
                document_name=last_doc["title"],
                document_text=last_doc["text"],
            )
            st.markdown("### Document Summary")
            st.write(summary)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_query = st.chat_input("Ask a question about your uploaded papers and searched papers")

    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        if not api_key:
            answer = "Gemini API key is missing. Add GEMINI_API_KEY in Streamlit secrets or paste it in the sidebar."
        else:
            with st.spinner("Thinking..."):
                try:
                    answer = generate_answer(
                        api_key=api_key,
                        model_name=model_name,
                        system_prompt=system_prompt,
                        context_text=st.session_state.combined_context,
                        chat_history=st.session_state.messages,
                        user_query=user_query,
                        documents=st.session_state.documents,
                        search_results=st.session_state.paper_search_results,
                        add_three_citations=add_three_citations,
                    )
                except Exception as e:
                    answer = f"An error occurred while generating the answer: {e}"

        st.session_state.last_answer = answer
        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)

# =========================
# FOOTER
# =========================
st.markdown("---")
st.caption("Built with Streamlit + Gemini + PDF/URL reading + Crossref search")
