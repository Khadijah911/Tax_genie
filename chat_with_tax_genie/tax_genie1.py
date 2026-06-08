

!pip install langchain==0.3.27 langchain-core==0.3.72 langchain-text-splitters==0.3.9 langchain-openai==0.2.12 langchain-community==0.3.26 chromadb

import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from openai import OpenAI
from google.colab import userdata
openai_api_key = userdata.get('OPENAI_API_KEY')
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    api_key=openai_api_key
)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "").strip())

pip install pypdf

from langchain_community.document_loaders import PyPDFLoader, TextLoader
import os

all_documents = []

# Load the PWC PDF
pwc_loader = PyPDFLoader('/content/the nigeria tax act pwc.pdf')
pwc_docs = pwc_loader.load()
for doc in pwc_docs:
    doc.metadata['source'] = 'PWC'
all_documents.extend(pwc_docs)
print(f" PWC: {len(pwc_docs)} pages")

# Load KPMG PDF
kpmg_loader = PyPDFLoader('/content/The Nigeria Tax Act (NTA), 2025 kpmg.pdf')
kpmg_docs = kpmg_loader.load()
for doc in kpmg_docs:
    doc.metadata['source'] = 'KPMG'
all_documents.extend(kpmg_docs)
print(f" KPMG: {len(kpmg_docs)} pages")

# Load Legit.ng text file
legit_loader = TextLoader('/content/legit.txt', encoding='utf-8')
legit_docs = legit_loader.load()
for doc in legit_docs:
    doc.metadata['source'] = 'Legit.ng'
all_documents.extend(legit_docs)
print(f" Legit.ng: {len(legit_docs)} document")

print(f"\n Total: {len(all_documents)} documents loaded")

from langchain.text_splitter import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100)
chunks = splitter.split_documents(all_documents)
print(f" {len(chunks)} chunks")

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

emb = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai_api_key)
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=emb,
    persist_directory="./chroma_db"
)
results = vectorstore.similarity_search("What is the tax rate for small companies?", k=1)
print(f"\n Test query result: {results[0].page_content[:150]}...")

from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

llm = ChatOpenAI(model_name="gpt-4o", temperature=0, api_key=openai_api_key)

template = """You are an expert on Nigerian tax law. Use the provided context to answer questions accurately.
CRITICAL INSTRUCTIONS:
1. ALWAYS specify which tax you're referring to (CIT, VAT, PIT, etc.)
2. NEVER mix thresholds between different taxes:
   - CIT small company exemption: ≤₦50M turnover
   - VAT small business exemption: ≤100M turnover
   - These are DIFFERENT thresholds for DIFFERENT taxes
3. For Personal Income Tax:
   - First ₦800,000 is tax-free (0% rate)
   - Progressive rates: 15%, 18%, 21%, 23%, 25%
4. ALWAYS cite your specific source (PWC page X, KPMG section Y, Legit.ng)
5. If context contains conflicting info, state BOTH with sources
6. For numerical calculations:
   - If question asks for specific tax amount, say: "I recommend using the calculator tab for accurate computation"
   - NEVER assume tax rates or bands - only use exact figures from context

 HALLUCINATION PREVENTION:
- If the context does NOT contain information about a topic, explicitly say: "The documents I have access to do not specify [X]. I cannot provide this information without risking inaccuracy."
- NEVER generalize statutory rates (like "medium companies pay 20%") unless explicitly stated in context
- NEVER invent tax categories (like "medium companies") that aren't in the source documents
- If asked about edge cases or unusual scenarios, acknowledge the limitation

IMPORTANT: When answering about "small companies", IMMEDIATELY clarify:
- "For CIT purposes, small companies have turnover ≤₦50M"
- "For VAT purposes, small businesses have turnover ≤₦100M"

For conceptual "Will X pay tax?" questions:
- Explain the relevant threshold/exemption
- State whether the scenario meets the criteria
- Do NOT redirect to calculator

Context from official tax documents:
{context}

Question: {question}

Detailed Answer:"""
prompt = PromptTemplate(template=template, input_variables=["context", "question"])

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True
)

def ask_refined(question):
    result = qa_chain.invoke({"query": question})
    print(f" QUESTION: {question}")
    print(f"\n ANSWER:\n{result['result']}")

    sources = {}
    for doc in result['source_documents']:
        source = doc.metadata['source']
        if source not in sources:
            sources[source] = []
        sources[source].append(doc.page_content[:100] + "...")

    print("SOURCES USED:")
    for source, snippets in sources.items():
        print(f"\n  {source}:")
        for snippet in snippets[:2]:  # Show max 2 snippets per source
            print(f" {snippet}")

    return result

ask_refined("What is the turnover threshold for a small company?")

ask_refined("What are the exact PIT tax bands and rates for 2026?")

ask_refined("Who pays Development Levy and at what rate?")

def calculate_pit(income, pension=0, nhf=0, nhis=0, life_insurance=0,
                  rent=0, is_military=False, is_pensioner=False):
    """
    Calculate Personal Income Tax (2026)

    Args:
        income: Annual gross income (₦)
        pension: Pension contributions (₦)
        nhf: National Housing Fund (₦)
        nhis: National Health Insurance (₦)
        life_insurance: Life insurance premiums (₦)
        rent: Annual rent paid (₦)
        is_military: Military officer (exempt)
        is_pensioner: Pensioner (exempt)
    """

    # Tax bands (2026)
    bands = [
        (800_000, 0.00),      # First 800k at 0%
        (2_200_000, 0.15),    # Next 2.2M at 15%
        (9_000_000, 0.18),    # Next 9M at 18%
        (13_000_000, 0.21),   # Next 13M at 21%
        (25_000_000, 0.23),   # Next 25M at 23%
        (float('inf'), 0.25)  # Above 50M at 25%
    ]

    #  exemptions
    if is_military:
        print(f"\n{'='*60}")
        print("EXEMPT - Military officer")
        print(f"Gross Income: ₦{income:,.2f}")
        print(f"Tax: ₦0.00")
        print(f"{'='*60}\n")
        return 0

    if is_pensioner:
        print(f"\n{'='*60}")
        print("EXEMPT - Pensioner")
        print(f"Gross Income: ₦{income:,.2f}")
        print(f"Tax: ₦0.00")
        print(f"{'='*60}\n")
        return 0

    #Calculate deductions
    rent_relief = min(rent * 0.20, 500_000)
    total_deductions = pension + nhf + nhis + life_insurance + rent_relief

    # Taxable income
    taxable = max(0, income - total_deductions)

    if taxable <= 800_000:
        print(f"\n{'='*60}")
        print(f"Gross Income:      ₦{income:,.2f}")
        print(f"Total Deductions:  ₦{total_deductions:,.2f}")
        print(f"Taxable Income:    ₦{taxable:,.2f}")
        print(f"\n EXEMPT - Below ₦800,000 threshold")
        print(f"Tax Payable:       ₦0.00")
        print(f"Net Income:        ₦{income - pension - nhf - nhis:,.2f}")
        print(f"{'='*60}\n")
        return 0

    #progressive tax
    remaining = taxable
    total_tax = 0
    cumulative = 0

    print(f"\n{'='*60}")
    print("PERSONAL INCOME TAX CALCULATION (2026)")
    print(f"{'='*60}")
    print(f"\nGross Income:      ₦{income:,.2f}")
    print(f"\nDEDUCTIONS:")
    print(f"  Pension:         ₦{pension:,.2f}")
    print(f"  NHF:             ₦{nhf:,.2f}")
    print(f"  NHIS:            ₦{nhis:,.2f}")
    print(f"  Life Insurance:  ₦{life_insurance:,.2f}")
    print(f"  Rent Relief:     ₦{rent_relief:,.2f}")
    print(f"  {'─'*40}")
    print(f"  Total:           ₦{total_deductions:,.2f}")
    print(f"\nTaxable Income:    ₦{taxable:,.2f}")
    print(f"\nTAX BREAKDOWN:")

    for limit, rate in bands:
        if remaining <= 0:
            break

        amount = min(remaining, limit)
        tax = amount * rate

        print(f"  ₦{cumulative:,.0f} - ₦{cumulative + amount:,.0f}")
        print(f"    @ {rate*100}% = ₦{tax:,.2f}")

        total_tax += tax
        remaining -= amount
        cumulative += amount

    net = income - pension - nhf - nhis - total_tax
    effective = (total_tax / income * 100) if income > 0 else 0

    print(f"\n{'─'*60}")
    print(f"Total Tax:         ₦{total_tax:,.2f}")
    print(f"Effective Rate:    {effective:.2f}%")
    print(f"Net Income:        ₦{net:,.2f}")
    print(f"{'='*60}\n")

    return total_tax

def check_company_tax(turnover, fixed_assets, is_professional=False):
    """
    Check company tax status and obligations

    Args:
        turnover: Annual gross turnover (₦)
        fixed_assets: Total fixed assets (₦)
        is_professional: Professional services company?
    """

    print(f"\n{'='*60}")
    print("COMPANY TAX STATUS CHECK")
    print(f"{'='*60}")
    print(f"\nTurnover:        ₦{turnover:,.2f}")
    print(f"Fixed Assets:    ₦{fixed_assets:,.2f}")
    print(f"Professional:    {'Yes' if is_professional else 'No'}")

    # Check if small company for CIT
    is_small_cit = (turnover <= 50_000_000 and
                    fixed_assets <= 250_000_000 and
                    not is_professional)

    # Check if small business for VAT
    is_small_vat = (turnover <= 100_000_000 and
                    fixed_assets <= 250_000_000 and
                    not is_professional)

    print(f"\n{'─'*60}")
    print("TAX OBLIGATIONS:")
    print(f"{'─'*60}")

    # CIT
    if is_small_cit:
        print(f"CIT: EXEMPT (Small company - turnover ≤ ₦50M)")
        cit_rate = 0
    else:
        print(f" CIT: 30% (Large company)")
        cit_rate = 30

    # VAT
    if is_small_vat:
        print(f" VAT: EXEMPT from filing (turnover ≤ ₦100M)")
    else:
        print(f" VAT: Must charge 7.5% and file returns")

    # Development Levy
    if is_small_cit:
        print(f" Development Levy: EXEMPT")
    else:
        print(f" Development Levy: 4% of assessable profit")

    print(f"{'='*60}\n")

    return {
        'is_small_cit': is_small_cit,
        'is_small_vat': is_small_vat,
        'cit_rate': cit_rate
    }

def chat_with_tax_genie(message, history):


    # Initialize history with welcome message if empty
    if not history:
        welcome = """Hi! I am **Tax_genie** 🇳🇬 and I am here to answer all your questions on tax in Nigeria!

You can ask me anything about:
- Personal Income Tax
- Company Income Tax
- VAT
- Development Levy
- Capital Gains Tax
- Tax exemptions and reliefs

What would you like to know?"""
        history = [[None, welcome]]
    # ROUTING LOGIC with IMPROVED INTENT DETECTION

    route_type, params = route_query(message)

    if route_type == 'calculator':
        # Detected explicit calculation request
        if not params:
            response = """I can help you calculate tax, but I need specific numbers.

**Please use the Calculator tabs** at the top:
- **Personal Income Tax tab** - for salary/personal income
- **Company Tax tab** - for business revenue

Or ask me to explain how taxes are calculated!"""
        else:
            # Detected numbers AND calculation intent
            amount = params[0]
            response = f"""I detected you want to calculate tax on **₦{amount:,.2f}**.

For **accurate calculations**, please use the dedicated calculator tabs.

**Why use the calculator?**
- It applies the exact progressive tax bands
- It handles all deductions correctly
- It accounts for exemptions automatically

I can explain the tax laws if you prefer!"""

    else:
        # Use RAG for informational/legal/conceptual queries
        result = qa_chain.invoke({"query": message})
        response = result['result']

        # Add detailed sources
        sources = set([doc.metadata['source'] for doc in result['source_documents']])
        response += f"\n\n *Sources: {', '.join(sources)}*"

        # Add warning if answer might mix thresholds
        if "50" in response and "100" in response and "million" in response.lower():
            response += "\n\n *Note: CIT exemption (₦50M) and VAT exemption (₦100M) are different thresholds.*"

        # Flag potential hallucination
        if "medium compan" in response.lower() and "20%" in response:
            response += "\n\n *Note: There is no 'medium company' category in Nigerian tax law. Companies are either small (≤₦50M) or large (>₦50M) for CIT purposes.*"

    history.append([message, response])

    return history, history

def calculate_pit_gradio(income, pension, nhf, nhis, life_insurance, rent, is_military, is_pensioner):
    """
    Wrapper for Gradio - formats calculator output as Markdown
    """

    # Capture the calculator output
    from io import StringIO
    import sys

    # Redirect print output to string
    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()

    # Run calculator
    calculate_pit(income, pension, nhf, nhis, life_insurance, rent, is_military, is_pensioner)

    # Get output
    output = mystdout.getvalue()
    sys.stdout = old_stdout

    # Format for Gradio
    return f"```\n{output}\n```"


def check_company_gradio(turnover, fixed_assets, is_professional):
    """
    Wrapper for Gradio - formats company checker output as Markdown
    """

    from io import StringIO
    import sys

    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()

    check_company_tax(turnover, fixed_assets, is_professional)

    output = mystdout.getvalue()
    sys.stdout = old_stdout

    return f"```\n{output}\n```"

print(" Gradio wrapper functions created!")

import re

def route_query(user_message):
    """
    Improved router that distinguishes:
    - Conceptual questions (use RAG)
    - Calculation requests (use calculator)

    Returns:
        ('calculator', params) or ('rag', None)
    """

    msg_lower = user_message.lower()

    # PRIORITY 1: Conceptual Questions (Even with Numbers)
    # These should ALWAYS use RAG, never calculator


    conceptual_patterns = [
        r'what (is|are) (the )?tax',
        r'explain.*tax',
        r'how (is|are).*tax.*calculated',
        r'how does.*tax.*work',
        r'definition of',
        r'what does.*mean',
        r'is.*exempt',
        r'will.*pay.*tax',
        r'do.*need to',
        r'qualify as',
        r'considered a',
        r'difference between',
        r'threshold for',
        r'obligations for',
        r'who pays',
        r'what rate',
        r'can.*claim',
        r'eligible for'
    ]

    for pattern in conceptual_patterns:
        if re.search(pattern, msg_lower):
            return ('rag', None)


    # PRIORITY 2: Explicit Calculation Requests
    # Only trigger calculator for EXPLICIT computation requests


    explicit_calc_patterns = [
        r'calculate (my|the) tax',
        r'how much tax (do i|will i|would i) (owe|pay)',
        r'what (is|would be) my tax',
        r'compute.*tax',
        r'work out.*tax',
        r'figure out.*tax'
    ]

    for pattern in explicit_calc_patterns:
        if re.search(pattern, msg_lower):
            return ('calculator', extract_numbers(user_message))

    # PRIORITY 3: Implicit Calculation with Personal Context
    # "I earn ₦X" or "My salary is ₦Y"


    personal_calc_patterns = [
        r'(i|my) (earn|make|get|have).*₦?\d',
        r'my (salary|income|wage).*₦?\d',
        r'(i|we) (have|got).*turnover.*₦?\d'
    ]

    for pattern in personal_calc_patterns:
        if re.search(pattern, msg_lower):
            return ('calculator', extract_numbers(user_message))


    # DEFAULT: Use RAG
    # If none of the above match, assume it's conceptual


    return ('rag', None)


def extract_numbers(text):
    """Extract all numbers from text"""
    numbers = re.findall(r'₦?\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)', text)
    return [float(n.replace(',', '')) for n in numbers] if numbers else []


# TEST

test_queries = [
    # Should route to RAG (conceptual)
    ("What is the CIT rate for small companies?", "rag"),
    ("Will a company with ₦35M turnover pay CIT?", "rag"),
    ("How is PIT calculated?", "rag"),
    ("Explain tax bands", "rag"),
    ("What's the threshold for VAT?", "rag"),

    # Should route to calculator
    ("How much tax will I pay on ₦3M salary?", "calculator"),
    ("My income is ₦10M, what's my tax?", "calculator"),
    ("I earn ₦500,000 monthly", "calculator")
]

print("🧪 TESTING IMPROVED ROUTER:\n")
for query, expected in test_queries:
    route, _ = route_query(query)
    status = "Y" if route == expected else "N"
    print(f"{status} '{query}' → {route} (expected: {expected})")

import gradio as gr

def create_gradio_app():
    """
    Creates and returns the Tax_genie Gradio application
    """

    custom_css = """
    .gradio-container {
        font-family: 'Arial', sans-serif;
    }
    .tab-nav button {
        font-size: 16px;
        font-weight: 600;
    }
    """

    with gr.Blocks(
        css=custom_css,
        title="Tax_genie - Nigerian Tax Assistant",
        theme=gr.themes.Soft()
    ) as demo:

        gr.Markdown("""
        # 🇳🇬 Ask Me Anything About Tax
        ### Powered by AI | Based on Nigeria Tax Act 2026
        """)

        with gr.Tabs():
            # TAB 1: CHAT

            with gr.Tab("💬 Chat with Tax_genie"):

                chatbot = gr.Chatbot(
                    value=[[None,
                        "Hi! I am **Tax_GENIE** 🇳🇬 and I am here to answer all your questions on tax in Nigeria! \n\n"
                        "You can ask me anything about:\n"
                        "- Personal Income Tax\n"
                        "- Company Income Tax\n"
                        "- VAT\n"
                        "- Development Levy\n"
                        "- Capital Gains Tax\n"
                        "- Tax exemptions and reliefs\n\n"
                        "What would you like to know?"
                    ]],
                    height=500,
                    show_label=False
                )

                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="Type your tax question here...",
                        show_label=False,
                        scale=9
                    )
                    submit = gr.Button("Send", variant="primary", scale=1)

                clear = gr.Button("Clear Chat")

                state = gr.State(chatbot.value)
                submit.click(chat_with_tax_genie, [msg, state], [chatbot, state])
                msg.submit(chat_with_tax_genie, [msg, state], [chatbot, state])

                submit.click(lambda: "", None, msg)
                msg.submit(lambda: "", None, msg)

                clear.click(
                    lambda: (
                        [[None, "Hi! I am **Tax_genie** 🇳🇬 and I am here to answer all your questions on tax in Nigeria! "]],
                        [[None, "Hi! I am **Tax_genie** 🇳🇬 and I am here to answer all your questions on tax in Nigeria!"]]
                    ),
                    None,
                    [chatbot, state]
                )

            # TAB 2: PERSONAL INCOME TAX
            with gr.Tab(" Personal Income Tax"):

                gr.Markdown("### Calculate Your Personal Income Tax (PAYE)")

                with gr.Row():
                    with gr.Column():
                        income = gr.Number(label="Annual Gross Income (₦)", value=3_000_000)
                        pension = gr.Number(label="Pension Contributions (₦)", value=0)
                        nhf = gr.Number(label="NHF Contributions (₦)", value=0)
                        nhis = gr.Number(label="NHIS Contributions (₦)", value=0)

                    with gr.Column():
                        life_insurance = gr.Number(label="Life Insurance Premium (₦)", value=0)
                        rent = gr.Number(label="Annual Rent Paid (₦)", value=0)
                        is_military = gr.Checkbox(label="I am a military officer")
                        is_pensioner = gr.Checkbox(label="I am a pensioner")

                calc_btn = gr.Button("Calculate Tax", variant="primary", size="lg")
                result = gr.Markdown()

                calc_btn.click(
                    calculate_pit_gradio,
                    [income, pension, nhf, nhis, life_insurance, rent, is_military, is_pensioner],
                    result
                )

            # TAB 3: COMPANY TAX
            with gr.Tab("🏢 Company Tax"):

                gr.Markdown("### Check Your Company's Tax Obligations")

                with gr.Row():
                    turnover = gr.Number(label="Annual Turnover (₦)", value=45_000_000)
                    fixed_assets = gr.Number(label="Total Fixed Assets (₦)", value=200_000_000)
                    is_prof = gr.Checkbox(label="Professional services company")

                check_btn = gr.Button("Check Tax Status", variant="primary", size="lg")
                company_result = gr.Markdown()

                check_btn.click(
                    check_company_gradio,
                    [turnover, fixed_assets, is_prof],
                    company_result
                )

        gr.Markdown("""
        ---
        **Disclaimer:** This tool provides general information based on the Nigeria Tax Act 2026.
        Always consult with a qualified tax professional for specific advice.
        """)

    return demo

app = create_gradio_app()
app.launch(debug=True)

