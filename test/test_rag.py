import importlib
import sys
import types
from pathlib import Path

import pytest


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


class FakeDoc:
    def __init__(self, page_content, source, page):
        self.page_content = page_content
        self.metadata = {"source": source, "page": page}


class FakeVectorDB:
    def __init__(self, docs):
        self.docs = docs

    def max_marginal_relevance_search(self, question, k, fetch_k):
        return self.docs


class FakeResponse:
    def __init__(self, text):
        self.text = text


class FakeModels:
    def __init__(self, response_text):
        self.response_text = response_text
        self.last_prompt = ""

    def generate_content(self, model, contents, config):
        self.last_prompt = contents
        return FakeResponse(self.response_text)


class FakeClient:
    def __init__(self, api_key=None, response_text="Exoplanets are planets outside our solar system."):
        self.models = FakeModels(response_text)


@pytest.fixture
def rag_module(monkeypatch):
    dotenv_module = types.ModuleType("dotenv")
    dotenv_module.load_dotenv = lambda: None

    types_module = types.ModuleType("google.genai.types")
    types_module.GenerateContentConfig = lambda **kwargs: kwargs

    genai_module = types.ModuleType("google.genai")
    genai_module.Client = FakeClient
    genai_module.types = types_module

    google_module = types.ModuleType("google")
    google_module.genai = genai_module

    config_module = types.ModuleType("config")
    config_module.embedding_model = object()
    config_module.CHROMA_DB_PATH = "unused"
    config_module.TOP_K = 5
    config_module.GEMINI_MODEL = "fake-model"

    class FakeChroma:
        def __init__(self, persist_directory, embedding_function):
            self.persist_directory = persist_directory
            self.embedding_function = embedding_function

        def max_marginal_relevance_search(self, question, k, fetch_k):
            return []

    langchain_chroma_module = types.ModuleType("langchain_chroma")
    langchain_chroma_module.Chroma = FakeChroma

    monkeypatch.setitem(sys.modules, "dotenv", dotenv_module)
    monkeypatch.setitem(sys.modules, "google", google_module)
    monkeypatch.setitem(sys.modules, "google.genai", genai_module)
    monkeypatch.setitem(sys.modules, "google.genai.types", types_module)
    monkeypatch.setitem(sys.modules, "config", config_module)
    monkeypatch.setitem(sys.modules, "langchain_chroma", langchain_chroma_module)

    sys.modules.pop("rag_chatbot", None)
    return importlib.import_module("rag_chatbot")


def test_answer_is_non_empty(rag_module):
    rag_module.vector_db = FakeVectorDB(
        [FakeDoc("Exoplanets orbit stars other than the Sun.", "docs/exoplanets.pdf", 0)]
    )
    rag_module.client = FakeClient(response_text="An exoplanet is a planet that orbits a star outside our solar system.")

    rag_response, _ = rag_module.answer_question("What is an exoplanet?")

    assert rag_response.answer.strip() != ""


def test_sources_are_returned(rag_module):
    rag_module.vector_db = FakeVectorDB(
        [
            FakeDoc("JWST has several scientific instruments.", "docs/jwst.pdf", 1),
            FakeDoc("JWST includes NIRCam and MIRI.", "docs/jwst.pdf", 2),
        ]
    )
    rag_module.client = FakeClient(response_text="JWST has multiple science instruments for imaging and spectroscopy.")

    rag_response, sources = rag_module.answer_question("How many instruments does JWST have?")

    assert sources
    assert "jwst" in sources
    assert rag_response.sources


def test_answer_length_is_reasonable(rag_module):
    rag_module.vector_db = FakeVectorDB(
        [FakeDoc("Habitability depends on temperature and liquid water.", "docs/habitable_zone.pdf", 0)]
    )
    rag_module.client = FakeClient(
        response_text=(
            "The habitable zone is the orbital region around a star where conditions can allow "
            "liquid water to exist on a planet's surface."
        )
    )

    rag_response, _ = rag_module.answer_question("What is the habitable zone?")

    assert 20 <= len(rag_response.answer.strip()) <= 1000


def test_unknown_query_is_handled_gracefully(rag_module):
    rag_module.vector_db = FakeVectorDB([])
    rag_module.client = FakeClient(
        response_text="I do not have enough context from the retrieved documents to answer that question."
    )

    rag_response, sources = rag_module.answer_question("What is qwertyunknownplanet?")

    assert rag_response.answer.strip() != ""
    assert sources == {}
    assert rag_response.sources == []


def test_multi_turn_history_is_included_in_prompt(rag_module):
    rag_module.vector_db = FakeVectorDB(
        [FakeDoc("JWST has four main science instruments.", "docs/jwst.pdf", 0)]
    )
    fake_client = FakeClient(response_text="JWST has four main instruments.")
    rag_module.client = fake_client

    history = [
        {"role": "user", "content": "Tell me about JWST."},
        {"role": "assistant", "content": "JWST is a space telescope used for infrared astronomy."},
    ]
    rag_module.answer_question("How many instruments does it have?", chat_history=history)

    prompt = fake_client.models.last_prompt
    assert "User: Tell me about JWST." in prompt
    assert "Assistant: JWST is a space telescope used for infrared astronomy." in prompt
    assert "How many instruments does it have?" in prompt