import streamlit as st
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import re
from docx import Document
import PyPDF2

# Access OpenAI API key from Streamlit secrets
llm = ChatOpenAI(model="gpt-4o-mini", api_key=st.secrets["OPENAI_API_KEY"])

class State(TypedDict):
    """Represents the state of the essay grading process."""
    essay: str
    relevance_score: float
    grammar_score: float
    structure_score: float
    depth_score: float
    final_score: float

def extract_score(content: str) -> float:
    """Extract the numeric score from the LLM's response."""
    match = re.search(r'Score:\s*(\d+(\.\d+)?)', content)
    if match:
        return float(match.group(1))
    raise ValueError(f"Could not extract score from: {content}")

# Essay evaluation functions
def check_relevance(state: State) -> State:
    prompt = ChatPromptTemplate.from_template(
        "Analyze the relevance of the following essay to the given topic. "
        "Provide a relevance score between 0 and 1. "
        "Your response should start with 'Score: ' followed by the numeric score, "
        "then provide your explanation.\n\nEssay: {essay}"
    )
    result = llm.invoke(prompt.format(essay=state["essay"]))
    try:
        state["relevance_score"] = extract_score(result.content)
    except ValueError as e:
        st.write(f"Error in check_relevance: {e}")
        state["relevance_score"] = 0.0
    return state

def check_grammar(state: State) -> State:
    prompt = ChatPromptTemplate.from_template(
        "Analyze the grammar and language usage in the following essay. "
        "Provide a grammar score between 0 and 1. "
        "Your response should start with 'Score: ' followed by the numeric score, "
        "then provide your explanation.\n\nEssay: {essay}"
    )
    result = llm.invoke(prompt.format(essay=state["essay"]))
    try:
        state["grammar_score"] = extract_score(result.content)
    except ValueError as e:
        st.write(f"Error in check_grammar: {e}")
        state["grammar_score"] = 0.0
    return state

def analyze_structure(state: State) -> State:
    prompt = ChatPromptTemplate.from_template(
        "Analyze the structure of the following essay. "
        "Provide a structure score between 0 and 1. "
        "Your response should start with 'Score: ' followed by the numeric score, "
        "then provide your explanation.\n\nEssay: {essay}"
    )
    result = llm.invoke(prompt.format(essay=state["essay"]))
    try:
        state["structure_score"] = extract_score(result.content)
    except ValueError as e:
        st.write(f"Error in analyze_structure: {e}")
        state["structure_score"] = 0.0
    return state

def evaluate_depth(state: State) -> State:
    prompt = ChatPromptTemplate.from_template(
        "Evaluate the depth of analysis in the following essay. "
        "Provide a depth score between 0 and 1. "
        "Your response should start with 'Score: ' followed by the numeric score, "
        "then provide your explanation.\n\nEssay: {essay}"
    )
    result = llm.invoke(prompt.format(essay=state["essay"]))
    try:
        state["depth_score"] = extract_score(result.content)
    except ValueError as e:
        st.write(f"Error in evaluate_depth: {e}")
        state["depth_score"] = 0.0
    return state

def calculate_final_score(state: State) -> State:
    state["final_score"] = (
        state["relevance_score"] * 0.3 +
        state["grammar_score"] * 0.2 +
        state["structure_score"] * 0.2 +
        state["depth_score"] * 0.3
    )
    return state

# Workflow setup
workflow = StateGraph(State)
workflow.add_node("check_relevance", check_relevance)
workflow.add_node("check_grammar", check_grammar)
workflow.add_node("analyze_structure", analyze_structure)
workflow.add_node("evaluate_depth", evaluate_depth)
workflow.add_node("calculate_final_score", calculate_final_score)
workflow.add_conditional_edges("check_relevance", lambda x: "check_grammar" if x["relevance_score"] > 0.5 else "calculate_final_score")
workflow.add_conditional_edges("check_grammar", lambda x: "analyze_structure" if x["grammar_score"] > 0.6 else "calculate_final_score")
workflow.add_conditional_edges("analyze_structure", lambda x: "evaluate_depth" if x["structure_score"] > 0.7 else "calculate_final_score")
workflow.add_conditional_edges("evaluate_depth", lambda x: "calculate_final_score")
workflow.set_entry_point("check_relevance")
workflow.add_edge("calculate_final_score", END)
app = workflow.compile()

# Text extraction function
def extract_text_from_file(uploaded_file):
    if uploaded_file.name.endswith('.pdf'):
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    elif uploaded_file.name.endswith('.docx'):
        doc = Document(uploaded_file)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text
    else:
        st.error("Unsupported file type. Please upload a PDF or Word document.")
        return None

# Streamlit App
st.title("Essay Grading App")
uploaded_file = st.file_uploader("Upload an essay as PDF or Word document", type=["pdf", "docx"])

if uploaded_file:
    essay_text = extract_text_from_file(uploaded_file)
    if essay_text:
        st.write("Essay Content:")
        st.write(essay_text)

        # Run grading process
        initial_state = State(
            essay=essay_text,
            relevance_score=0.0,
            grammar_score=0.0,
            structure_score=0.0,
            depth_score=0.0,
            final_score=0.0
        )
        result = app.invoke(initial_state)

        # Display results
        st.write("### Grading Results")
        st.write(f"Final Essay Score: {result['final_score']:.2f}")
        st.write(f"Relevance Score: {result['relevance_score']:.2f}")
        st.write(f"Grammar Score: {result['grammar_score']:.2f}")
        st.write(f"Structure Score: {result['structure_score']:.2f}")
        st.write(f"Depth Score: {result['depth_score']:.2f}")
