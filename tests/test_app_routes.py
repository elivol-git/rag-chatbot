"""Route-level checks that need no Ollama and no vector store."""

import pytest

from src import app as app_module


@pytest.fixture
def client():
    app_module.app.config.update(TESTING=True)
    return app_module.app.test_client()


def test_health_reports_configuration(client):
    body = client.get("/api/health").get_json()
    assert body["embed_model"] and body["llm_model"]
    assert body["status"] in {"ok", "degraded"}


def test_ask_requires_a_question(client):
    assert client.post("/api/ask", json={"question": "  "}).status_code == 400
    assert client.post("/api/ask/stream", json={}).status_code == 400


@pytest.mark.skipif(
    not (app_module.FRONTEND_DIST / "index.html").is_file(),
    reason="frontend/dist not built",
)
class TestBuiltFrontend:
    def test_index_is_served(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert b"<div id=\"root\">" in response.data

    def test_hashed_assets_are_served(self, client):
        """Regression: a Windows backslash path made send_from_directory 404."""
        assets = list((app_module.FRONTEND_DIST / "assets").glob("*.js"))
        assert assets, "expected a built JS bundle"

        response = client.get(f"/assets/{assets[0].name}")
        assert response.status_code == 200
        assert len(response.data) > 1000
        assert b"<!doctype html>" not in response.data[:200].lower()

    def test_unknown_path_falls_back_to_the_spa(self, client):
        response = client.get("/some/client/route")
        assert response.status_code == 200
        assert b"<div id=\"root\">" in response.data
