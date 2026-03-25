import os
import re
import io
from typing import List, Dict

import requests
import streamlit as st
from bs4 import BeautifulSoup
from pypdf import PdfReader
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="ScholarGPT Pro",
    page_icon="📚",
    layout="wide",
)

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
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
.citation-box {
    background: #0f172a;
    border-left: 4px solid #38bdf8;
    padding: 10px 12px;
    border-radius: 10px;
    margin: 10px 0;
}
hr {
    border-color: rgba(255,255,255,0.08) !important;
}
</style>
""", unsafe_allow_html=True)

# =========================
# CONSTANTS
# =========================
APP_TITLE = "📚 ScholarGPT Pro"
APP_SUBTITLE = "Upload papers, search many papers, add links, and chat with citations."

DEFAULT_SYSTEM_PROMPT = """
You are a research assistant specializing in academic writing with APA 7th edition formatting.

RESPONSE FORMAT:
1. Start your answer directly without mentioning paper names or sources at the beginning
2. Write in clear academic prose with proper APA 7th edition in-text citations (Author, Year)
3. Provide a "References" section at the end with proper APA 7th edition formatting

CITATION RULES:
- Use (Author, Year) format for in-text citations
- For multiple authors: (Smith & Johnson, 2023) for 2 authors, (Smith et al., 2023) for 3+ authors
- If no author: (Organization, Year) or ("Title of Work", Year)
- For direct quotes: include page numbers (Author, Year, p. 23)

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
"""

USER_AGENT = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

DEFAULT_MODEL = "gemini-2.5-flash"

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

# =========================
# HELPERS
# =========================
def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_pdf_text_from_file(uploaded_file) -> str:
    try:
        pdf = PdfReader(uploaded_file)
        pages = []
        for page in pdf.pages:
            pages.append(page.extract_text() or "")
        return clean_text("\n".join(pages))
    except Exception as e:
        return f"ERROR_READING_PDF_FILE: {e}"


def extract_pdf_text_from_url(url: str) -> str:
    try:
        response = requests.get(url, headers=USER_AGENT, timeout=40)
        response.raise_for_status()
        file_like = io.BytesIO(response.content)
        pdf = PdfReader(file_like)
        pages = []
        for page in pdf.pages:
            pages.append(page.extract_text() or "")
        return clean_text("\n".join(pages))
    except Exception as e:
        return f"ERROR_READING_PDF_URL: {e}"


def extract_html_text_from_url(url: str) -> str:
    try:
        response = requests.get(url, headers=USER_AGENT, timeout=40)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
            tag.decompose()

        title = soup.title.string.strip() if soup.title and soup.title.string else ""
        meta_description = ""
        meta_abs = ""

        desc = soup.find("meta", attrs={"name": "description"})
        if desc and desc.get("content"):
            meta_description = desc["content"].strip()

        abs_tag = soup.find("meta", attrs={"name": "citation_abstract"})
        if abs_tag and abs_tag.get("content"):
            meta_abs = abs_tag["content"].strip()

        body_text = clean_text(soup.get_text(separator=" ", strip=True))
        return clean_text(
            f"Title: {title}\nDescription: {meta_description}\nAbstract: {meta_abs}\nBody: {body_text}"
        )
    except Exception as e:
        return f"ERROR_READING_HTML_URL: {e}"


def extract_text_from_url(url: str) -> str:
    url = url.strip()
    if url.lower().endswith(".pdf"):
        return extract_pdf_text_from_url(url)
    return extract_html_text_from_url(url)


def build_combined_context(documents: List[Dict], search_results: List[Dict], max_chars: int = 140000) -> str:
    if not documents and not search_results:
        return ""
    
    parts = []
    total = 0
    
    # Add uploaded documents (full text available)
    for doc in documents:
        block = f"\n\n===== UPLOADED DOCUMENT: {doc['name']} =====\n{doc['text']}\n"
        if total + len(block) > max_chars:
            remaining = max_chars - total
            if remaining > 0:
                parts.append(block[:remaining])
            break
        parts.append(block)
        total += len(block)
    
    # Add search results (metadata only, formatted for academic use)
    for paper in search_results:
        block = f"\n\n===== SEARCHED PAPER: {paper['title']} =====\n"
        block += f"Authors: {paper['authors']}\n"
        block += f"Year: {paper['year']}\n"
        block += f"Journal: {paper['journal']}\n"
        block += f"DOI: {paper['doi']}\n"
        block += f"Link: {paper['link']}\n\n"
        block += f"ACADEMIC NOTE: This is a paper reference found through search. Use this metadata for citations and to answer questions about the paper's existence, authors, publication details, and research topics. The full text content is not available, but you can cite this paper using the provided information in APA 7th edition format.\n"
        
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


def search_crossref_papers(query: str, rows: int = 50) -> List[Dict]:
    try:
        url = "https://api.crossref.org/works"
        params = {
            "query": query,
            "rows": rows,
            "select": "title,URL,DOI,author,issued,container-title,type"
        }
        r = requests.get(url, params=params, headers=USER_AGENT, timeout=40)
        r.raise_for_status()
        data = r.json()

        items = data.get("message", {}).get("items", [])
        results = []

        for item in items:
            title = item["title"][0] if item.get("title") else "Untitled"
            journal = item["container-title"][0] if item.get("container-title") else ""
            doi = item.get("DOI", "")
            link = item.get("URL", "")
            year = ""

            issued = item.get("issued", {})
            date_parts = issued.get("date-parts", [])
            if date_parts and date_parts[0]:
                year = str(date_parts[0][0])

            authors = []
            for a in item.get("author", []):
                given = a.get("given", "")
                family = a.get("family", "")
                fullname = f"{given} {family}".strip()
                if fullname:
                    authors.append(fullname)

            results.append({
                "title": title,
                "journal": journal,
                "year": year,
                "doi": doi,
                "link": link,
                "authors": ", ".join(authors[:8]) if authors else "N/A"
            })

        return results
    except Exception as e:
        st.error(f"Paper search failed: {e}")
        return []


def build_source_list(documents: List[Dict], search_results: List[Dict]) -> str:
    if not documents and not search_results:
        return "No loaded sources."
    
    lines = []
    
    # Uploaded documents
    for i, doc in enumerate(documents, start=1):
        lines.append(f"[{i}] {doc['name']} ({doc['source']}) - UPLOADED")
    
    # Search results
    for i, paper in enumerate(search_results, start=len(documents) + 1):
        lines.append(f"[{i}] {paper['title']} ({paper['authors']}, {paper['year']}) - SEARCHED")
    
    return "\n".join(lines)


def generate_answer(
    api_key: str,
    model_name: str,
    system_prompt: str,
    context_text: str,
    chat_history: List[Dict],
    user_query: str,
    documents: List[Dict],
    search_results: List[Dict],
    add_three_citations: bool = False,  # Disabled - using APA 7th edition instead
) -> str:
    configure_gemini(api_key)
    model = genai.GenerativeModel(model_name)

    recent_history = chat_history[-10:] if chat_history else []
    history_text = ""
    for msg in recent_history:
        role = msg.get("role", "user").upper()
        content = msg.get("content", "")
        history_text += f"{role}: {content}\n"

    source_list = build_source_list(documents, search_results)

    prompt = f"""
{system_prompt}

AVAILABLE SOURCES:
{source_list}

DOCUMENT CONTEXT:
{context_text if context_text else "No document context is loaded yet."}

CHAT HISTORY:
{history_text if history_text else "No previous chat history."}

USER QUESTION:
{user_query}

Write the answer clearly.
Use only the provided sources where possible.
For uploaded documents, you have full text content.
For searched papers, you have metadata (title, authors, journal, year) but not full text.
If the answer is not in the provided sources, say so clearly.
Mention source document names naturally in the answer where relevant.
Do not invent author names, years, or references.
"""

    response = model.generate_content(prompt)
    answer = (response.text or "").strip()
    if not answer:
        answer = "No response was returned."

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
manual_api_key = st.sidebar.text_input(
    "Gemini API Key (optional)",
    type="password",
    value=api_key,  # Auto-fill with existing key
    help="Leave empty to use the key from .env file"
)

if manual_api_key.strip() and manual_api_key != api_key:
    api_key = manual_api_key.strip()

model_name = st.sidebar.selectbox(
    "Gemini Model",
    options=["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.5-pro"],
    index=0,
)

system_prompt = st.sidebar.text_area(
    "System Prompt",
    value=DEFAULT_SYSTEM_PROMPT,
    height=220,
)

search_rows = st.sidebar.slider(
    "Number of papers to fetch",
    min_value=10,
    max_value=100,
    value=50,
    step=10,
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
# HEADER
# =========================
st.markdown(f'<div class="main-title">{APP_TITLE}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sub-title">{APP_SUBTITLE}</div>', unsafe_allow_html=True)

if not api_key:
    st.warning("⚠️ No Gemini API key found. Please add GEMINI_API_KEY to your .env file or enter it in the sidebar.")
else:
    st.success("✅ Gemini API key loaded successfully")

tab1, tab2, tab3 = st.tabs(["📄 Upload Papers", "🔗 Add Paper Link / Search Papers", "💬 Chat With Papers"])

# =========================
# TAB 1
# =========================
with tab1:
    st.subheader("Upload PDF Papers")
    uploaded_files = st.file_uploader(
        "Upload one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        if st.button("Process Uploaded PDFs"):
            new_docs = []
            with st.spinner("Reading uploaded PDFs..."):
                for file in uploaded_files:
                    text = extract_pdf_text_from_file(file)
                    if text and not text.startswith("ERROR_READING_PDF_FILE"):
                        new_docs.append({
                            "name": file.name,
                            "source": "uploaded_pdf",
                            "text": text,
                        })
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
            st.markdown(f'<div class="info-card"><b>Loaded documents:</b> {len(st.session_state.documents)}</div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="info-card"><b>Searched papers:</b> {len(st.session_state.paper_search_results)}</div>', unsafe_allow_html=True)

        for i, doc in enumerate(st.session_state.documents, start=1):
            with st.expander(f"{i}. {doc['name']}"):
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
                if extracted.startswith("ERROR_READING_"):
                    st.error(extracted)
                else:
                    name_guess = paper_url.split("/")[-1] or "paper_link"
                    st.session_state.documents.append({
                        "name": name_guess,
                        "source": paper_url,
                        "text": extracted,
                    })
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
            with st.spinner("Searching papers..."):
                results = search_crossref_papers(search_query, rows=search_rows)
                st.session_state.paper_search_results = results
                st.session_state.combined_context = build_combined_context(
                    st.session_state.documents, st.session_state.paper_search_results
                )

    if st.session_state.paper_search_results:
        st.markdown(f"### Search Results ({len(st.session_state.paper_search_results)})")
        for idx, item in enumerate(st.session_state.paper_search_results, start=1):
            st.markdown('<div class="paper-card">', unsafe_allow_html=True)
            st.markdown(f"**{idx}. {item['title']}**")
            st.markdown(f"<div class='small-muted'><b>Authors:</b> {item['authors']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='small-muted'><b>Journal:</b> {item['journal'] or 'N/A'}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='small-muted'><b>Year:</b> {item['year'] or 'N/A'}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='small-muted'><b>DOI:</b> {item['doi'] or 'N/A'}</div>", unsafe_allow_html=True)
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
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            f'<div class="info-card"><b>Searched papers:</b> {len(st.session_state.paper_search_results)}</div>',
            unsafe_allow_html=True
        )

    if st.session_state.documents and api_key:
        if st.button("Summarize Last Added Document"):
            last_doc = st.session_state.documents[-1]
            with st.spinner("Summarizing document..."):
                summary = summarize_document(
                    api_key=api_key,
                    model_name=model_name,
                    document_name=last_doc["name"],
                    document_text=last_doc["text"],
                )
                st.markdown("### Document Summary")
                st.write(summary)

    # Show chat history
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
