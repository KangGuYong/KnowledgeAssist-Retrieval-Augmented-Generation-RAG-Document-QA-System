"""VectorStoreService.delete_by_document_id must actually delete.

Runs against a real Chroma instance (FakeEmbeddings, no network/model) rather
than a mock, because what this guards lives in the third-party wrapper's
plumbing: whether Chroma.delete() forwards `where=` to the underlying chromadb
collection at all. langchain-community 0.0.38 did not - it passed only `ids=`
and dropped the rest, so every delete raised "You must provide either ids,
where, or where_document to delete." That regression was found while
smoke-testing the 0.1.20/0.0.38 bump and worked around by calling
`_collection` directly; 0.4.2 forwards **kwargs, so the workaround is gone and
this test is what keeps its removal honest.
"""

import pytest
from langchain_core.documents import Document
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
