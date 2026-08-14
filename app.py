import os
import json
import gradio as gr
import google.generativeai as genai
from tavily import TavilyClient
from dotenv import load_dotenv

# LOAD ENVIRONMENT VARIABLES
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# VALIDATE API KEYS
if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY is not set. "
        "Please add it to your .env file."
    )
if not TAVILY_API_KEY:
    raise ValueError(
        "TAVILY_API_KEY is not set. "
        "Please add it to your .env file."
    )

# CONFIGURE APIS
genai.configure(api_key=GEMINI_API_KEY)
tavily = TavilyClient(
    api_key=TAVILY_API_KEY
)

# FIND A WORKING GEMINI MODEL
def get_working_model():
    candidates = []
    try:
        for m in genai.list_models():
            methods = getattr(
                m,
                "supported_generation_methods",
                []
            )
            if "generateContent" in methods:
                candidates.append(m.name)
    except Exception as e:
        raise RuntimeError(
            f"Unable to retrieve Gemini models: {str(e)}"
        )
    if not candidates:
        raise RuntimeError(
            "No Gemini models with generateContent support "
            "were found for this API key."
        )
    # Preferred models
    preferred_keywords = [
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-1.5-flash",
        "flash",
        "pro"
    ]
    # Select preferred model
    for keyword in preferred_keywords:
        for name in candidates:
            if keyword in name.lower():
                return name, candidates
    # Fallback
    return candidates[0], candidates

# INITIALIZE GEMINI MODEL
MODEL_NAME, ALL_MODELS = get_working_model()
model = genai.GenerativeModel(
    MODEL_NAME
)
print("=" * 60)
print("Research Assistant Started")
print("=" * 60)
print("Using Gemini model:", MODEL_NAME)
print("\nAvailable generateContent models:")
for name in ALL_MODELS[:20]:
    print("-", name)
print("=" * 60)

# HELPER FUNCTIONS
def clean_json_text(text):
    if not text:
        return ""
    text = text.strip()
    # Remove markdown JSON fences
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()

# GEMINI REQUEST
def ask_gemini(prompt):
    response = model.generate_content(
        prompt
    )
    # Normal response
    text = getattr(
        response,
        "text",
        None
    )
    if text:
        return text.strip()
    # Fallback response extraction
    parts = []
    try:
        for candidate in response.candidates:
            for part in candidate.content.parts:
                if hasattr(part, "text") and part.text:
                    parts.append(
                        part.text
                    )
    except Exception:
        pass
    return "\n".join(parts).strip()

# TAVILY WEB SEARCH
def search_sources(
    query,
    max_results=5
):
    response = tavily.search(
        query=query,
        search_depth="advanced",
        max_results=int(max_results),
        include_answer=False,
        include_raw_content=False
    )
    sources = []
    for idx, item in enumerate(
        response.get("results", []),
        start=1
    ):
        sources.append({
            "title": item.get(
                "title",
                "Untitled Source"
            ),
            "url": item.get(
                "url",
                ""
            ),
            "snippet": item.get(
                "content",
                ""
            )[:1000],
            "citation_label": f"[{idx}]"
        })
    return sources

# BUILD SOURCE CONTEXT
def build_source_context(
    sources
):
    blocks = []
    for idx, source in enumerate(
        sources,
        start=1
    ):
        blocks.append(
            f"Source [{idx}]\n"
            f"Title: {source['title']}\n"
            f"URL: {source['url']}\n"
            f"Snippet: {source['snippet']}\n"
        )
    return "\n".join(blocks)

# GENERATE CITED ANSWER
def generate_cited_answer(
    query,
    sources
):
    source_context = build_source_context(
        sources
    )
    prompt = f"""
You are a careful research assistant.
Your task is to answer the user's research question
using ONLY the provided sources.
RULES:
1. Use only information supported by the provided sources.
2. Every factual claim must include a citation.
3. Use citation format such as [1], [2], [1][3].
4. Do not invent facts.
5. Do not invent sources.
6. If the sources do not provide enough evidence, clearly say so.
7. If sources disagree, mention the disagreement.
8. Keep the answer concise and useful.
9. Do not create citations that do not correspond to the provided sources.
10. Prefer direct evidence from the sources.
USER QUERY:
{query}
PROVIDED SOURCES:
{source_context}
Write the final research answer now.
"""
    return ask_gemini(
        prompt
    )

# EXTRACT FACTUAL CLAIMS
def extract_claims(
    answer
):
    prompt = f"""
Extract up to 5 important factual claims from the
answer below.
Return JSON ONLY.
Use exactly this format:
{{
    "claims": [
        "claim 1",
        "claim 2"
    ]
}}
Do not include markdown.
Do not include explanations outside the JSON.
ANSWER:
{answer}
"""
    try:
        text = clean_json_text(
            ask_gemini(prompt)
        )
        data = json.loads(
            text
        )
        return [
            str(x)
            for x in data.get(
                "claims",
                []
            )
        ][:5]
    except Exception:
        return []

# VERIFY CLAIMS
def verify_claims(
    claims,
    sources
):
    if not claims:
        return []
    source_context = build_source_context(
        sources
    )
    prompt = f"""
You are a fact-checking assistant.
Verify each claim using ONLY the provided sources.
For each claim determine one of:
- Supported
- Partially Supported
- Insufficient Evidence
- Contradicted
Return JSON ONLY.
Use exactly this structure:
{{
    "claims": [
        {{
            "claim": "...",
            "verdict": "Supported",
            "explanation": "...",
            "citations": ["[1]", "[2]"]
        }}
    ]
}}
Rules:
1. Do not invent evidence.
2. Do not invent citations.
3. Citations must correspond to the provided sources.
4. If the source only partially supports a claim, use Partially Supported.
5. If there is not enough evidence, use Insufficient Evidence.
6. If the sources directly disagree with the claim, use Contradicted.
CLAIMS:
{json.dumps(
    claims,
    ensure_ascii=False
)}
SOURCES:
{source_context}
"""
    try:
        text = clean_json_text(
            ask_gemini(prompt)
        )
        data = json.loads(
            text
        )
        return data.get(
            "claims",
            []
        )
    except Exception:
        return []

# HTML SOURCE FORMATTER
def format_sources_html(
    sources
):
    if not sources:
        return (
            "<p>No sources found.</p>"
        )
    html = ""
    for source in sources:
        title = source.get(
            "title",
            "Untitled Source"
        )
        url = source.get(
            "url",
            ""
        )
        snippet = source.get(
            "snippet",
            ""
        )
        citation = source.get(
            "citation_label",
            ""
        )
        html += f"""
        <div style="
            background:#ffffff;
            border:1px solid #e5e7eb;
            border-radius:16px;
            padding:14px;
            margin-bottom:12px;
            box-shadow:0 2px 8px rgba(0,0,0,0.04);
        ">
            <div style="
                font-size:16px;
                font-weight:700;
                margin-bottom:8px;
                color:#111827;
            ">
                {citation} {title}
            </div>
            <div style="
                font-size:14px;
                color:#374151;
                line-height:1.55;
                margin-bottom:10px;
            ">
                {snippet}
            </div>
            <a
                href="{url}"
                target="_blank"
                style="
                    font-size:14px;
                    color:#0b57d0;
                    text-decoration:none;
                "
            >
                Open source →
            </a>
        </div>
        """
    return html

# HTML CLAIM VERIFICATION FORMATTER
def format_claims_html(
    claims
):
    if not claims:
        return (
            "<p>No claim verification available.</p>"
        )
    color_map = {
        "Supported":
            "#15803d",
        "Partially Supported":
            "#b45309",
        "Insufficient Evidence":
            "#6b7280",
        "Contradicted":
            "#b91c1c"
    }
    html = ""
    for item in claims:
        verdict = item.get(
            "verdict",
            "Insufficient Evidence"
        )
        color = color_map.get(
            verdict,
            "#6b7280"
        )
        citations = ", ".join(
            item.get(
                "citations",
                []
            )
        )
        if not citations:
            citations = "None"
        claim = item.get(
            "claim",
            ""
        )
        explanation = item.get(
            "explanation",
            ""
        )
        html += f"""
        <div style="
            background:#ffffff;
            border:1px solid #e5e7eb;
            border-radius:16px;
            padding:14px;
            margin-bottom:12px;
            box-shadow:0 2px 8px rgba(0,0,0,0.04);
        ">
            <div style="
                font-size:16px;
                font-weight:700;
                margin-bottom:8px;
                color:#111827;
            ">
                {claim}
            </div>
            <div style="
                margin-bottom:8px;
                font-size:14px;
            ">
                Verdict:
                <span style="
                    font-weight:700;
                    color:{color};
                ">
                    {verdict}
                </span>
            </div>
            <div style="
                font-size:14px;
                color:#374151;
                line-height:1.55;
                margin-bottom:8px;
            ">
                {explanation}
            </div>
            <div style="
                font-size:13px;
                color:#6b7280;
            ">
                Citations: {citations}
            </div>
        </div>
        """
    return html

# MAIN APPLICATION LOGIC
def run_research(
    query,
    max_results
):
    # Validate query
    if not query or not query.strip():
        return (
            "Please enter a research question.",
            "<p>No sources yet.</p>",
            "<p>No claim verification yet.</p>"
        )
    try:
        # SEARCH WEB
        sources = search_sources(
            query,
            max_results
        )
        if not sources:
            return (
                "No sources were found for this query.",
                "<p>No sources found.</p>",
                "<p>No claim verification available.</p>"
            )
        # GENERATE ANSWER
        answer = generate_cited_answer(
            query,
            sources
        )
        # EXTRACT CLAIMS
        claims = extract_claims(
            answer
        )
        # VERIFY CLAIMS
        verified_claims = verify_claims(
            claims,
            sources
        )
        # RETURN RESULTS
        return (
            answer,
            format_sources_html(
                sources
            ),
            format_claims_html(
                verified_claims
            )
        )
    except Exception as e:
        return (
            f"""
            ### Error
            `{str(e)}`
            """,
            "<p>Failed to retrieve sources.</p>",
            "<p>Failed to verify claims.</p>"
        )

# CSS
css = """
.gradio-container {
    max-width: 1280px !important;
    margin: auto !important;
}
.main-title {
    text-align: center;
    font-size: 34px;
    font-weight: 800;
    margin-bottom: 8px;
    color: #111827;
}
.sub-title {
    text-align: center;
    font-size: 16px;
    color: #4b5563;
    margin-bottom: 24px;
}
"""

# GRADIO APPLICATION
with gr.Blocks(
    theme=gr.themes.Soft(),
    css=css
) as demo:
    # HEADER
    gr.HTML(
        f"""
        <div class="main-title">
            Research Assistant with Citations
        </div>
        <div class="sub-title">
            Search the web, generate a cited answer,
            and verify factual claims.
            <br>
            <b>Active model:</b>
            {MODEL_NAME}
        </div>
        """
    )
    # INPUT SECTION
    with gr.Row():
        # QUERY
        with gr.Column(
            scale=2
        ):
            query_input = gr.Textbox(
                label="Research Question",
                placeholder=(
                    "Example: What are the latest "
                    "trends in AI agents for academic "
                    "research workflows?"
                ),
                lines=8
            )
            # NUMBER OF SOURCES
            max_results_input = gr.Slider(
                minimum=3,
                maximum=10,
                value=5,
                step=1,
                label="Number of sources"
            )
            # RUN BUTTON
            run_button = gr.Button(
                "Run Research",
                variant="primary"
            )
        # INSTRUCTIONS
        with gr.Column(
            scale=1
        ):
            gr.Markdown(
                """
                ### Instructions
                1. Enter a focused research question.
                2. Choose the number of sources.
                3. Click **Run Research**.
                4. Review the cited answer.
                5. Open the original sources.
                6. Review claim verification.
                """
            )

    # OUTPUT SECTION
    with gr.Row():
        # ANSWER
        with gr.Column(
            scale=2
        ):
            answer_output = gr.Markdown(
                label="Cited Answer"
            )
        # SOURCES
        with gr.Column(
            scale=1
        ):
            sources_output = gr.HTML(
                label="Sources"
            )
    # CLAIM VERIFICATION
    claims_output = gr.HTML(
        label="Claim Verification"
    )
    # BUTTON EVENT
    run_button.click(
        fn=run_research,
        inputs=[
            query_input,
            max_results_input
        ],
        outputs=[
            answer_output,
            sources_output,
            claims_output
        ]
    )

# LAUNCH APPLICATION
if __name__ == "__main__":
    demo.launch(
        share=True
    )
