"""VectorStoreService.delete_by_document_id must actually delete.

Regression found while smoke-testing the langchain 0.1.20/langchain-community
0.0.38 bump (see commit "bump langchain to add langchain-experimental"):
Chroma.delete() in langchain-community 0.0.38 only forwards `ids=` to the
underlying chromadb collection and silently drops `where=` into an unused
**kwargs, so `delete(where={"document_id": ...})` raised "You must provide
either ids, where, or where_document to delete." on every call. Confirmed via
a real Chroma instance (FakeEmbeddings, no network/model), not a mock, since
the bug lives in the third-party wrapper's plumbing.
"""

import pytest
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import FakeEmbeddings

from app.services.vector_store import VectorStoreService


@pytest.fixture
def service(tmp_path):
    vector_store = Chroma(
        collection_name="test_delete_by_document_id",
        embedding_function=FakeEmbeddings(size=8),
        persist_directory=str(tmp_path / "chroma"),
    )
    svc = VectorStoreService.__new__(VectorStoreService)
    svc.vector_store = vector_store
    return svc


def test_delete_by_document_id_removes_only_that_document(service):
    service.add_documents([Document(page_content="a")], "doc_1")
    service.add_documents([Document(page_content="b")], "doc_2")

    service.delete_by_document_id("doc_1")

    remaining_ids = [m["document_id"] for m in service.vector_store.get()["metadatas"]]
    assert remaining_ids == ["doc_2"]


def test_delete_by_document_id_is_a_noop_for_unknown_id(service):
    service.add_documents([Document(page_content="a")], "doc_1")

    service.delete_by_document_id("doc_never_uploaded")

    remaining_ids = [m["document_id"] for m in service.vector_store.get()["metadatas"]]
    assert remaining_ids == ["doc_1"]
