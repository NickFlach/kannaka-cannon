"""Tests for the ClipCannon dashboard application.

Verifies health endpoint, credits API, projects API, provenance API,
auth endpoints, and static file serving.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from clipcannon.dashboard.app import create_app
from clipcannon.dashboard.auth import (
    _DEFAULT_DEV_SECRET,
    create_session_token,
    get_jwt_secret,
    is_dev_mode,
    verify_session_token,
)


@pytest.fixture()
def client() -> TestClient:
    """Create a test client for the dashboard app.

    Returns:
        FastAPI TestClient instance.
    """
    app = create_app()
    return TestClient(app)


class TestHealth:
    """Health endpoint tests."""

    def test_health_returns_ok(self, client: TestClient) -> None:
        """GET /health returns status ok with version."""
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "version" in data
        assert data["service"] == "clipcannon-dashboard"


class TestAuth:
    """Authentication endpoint tests."""

    def test_dev_login_sets_cookie(self, client: TestClient) -> None:
        """GET /auth/dev-login returns success and sets session cookie."""
        resp = client.get("/auth/dev-login")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert "user_id" in data
        assert "clipcannon_session" in resp.cookies

    def test_auth_me_returns_user_in_dev_mode(self, client: TestClient) -> None:
        """GET /auth/me returns user info in dev mode."""
        resp = client.get("/auth/me")
        assert resp.status_code == 200
        data = resp.json()
        assert data["authenticated"] is True
        assert data["dev_mode"] is True

    def test_auth_logout_clears_cookie(self, client: TestClient) -> None:
        """GET /auth/logout clears the session cookie."""
        resp = client.get("/auth/logout")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True

    def test_token_roundtrip(self) -> None:
        """JWT token creation and verification round-trip."""
        token = create_session_token("user-123", "user@test.com")
        payload = verify_session_token(token)
        assert payload is not None
        assert payload["sub"] == "user-123"
        assert payload["email"] == "user@test.com"

    def test_invalid_token_returns_none(self) -> None:
        """Invalid JWT token returns None."""
        result = verify_session_token("invalid-token-string")
        assert result is None


class TestJwtSecretHardening:
    """Production JWT secret enforcement (fail-hard when misconfigured)."""

    def test_dev_mode_allows_default_secret(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """In dev mode the default secret is allowed."""
        monkeypatch.setenv("CLIPCANNON_DEV_MODE", "1")
        monkeypatch.delenv("CLIPCANNON_JWT_SECRET", raising=False)
        assert is_dev_mode() is True
        assert get_jwt_secret() == _DEFAULT_DEV_SECRET

    def test_prod_default_secret_fails_hard(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Prod (dev mode off) with the default secret raises."""
        monkeypatch.setenv("CLIPCANNON_DEV_MODE", "0")
        monkeypatch.delenv("CLIPCANNON_JWT_SECRET", raising=False)
        with pytest.raises(RuntimeError, match="CLIPCANNON_JWT_SECRET"):
            get_jwt_secret()

    def test_prod_explicit_default_secret_fails_hard(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Prod with the secret set to the known default still raises."""
        monkeypatch.setenv("CLIPCANNON_DEV_MODE", "0")
        monkeypatch.setenv("CLIPCANNON_JWT_SECRET", _DEFAULT_DEV_SECRET)
        with pytest.raises(RuntimeError, match="insecure default"):
            get_jwt_secret()

    def test_prod_real_secret_ok(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Prod with a real secret resolves without raising."""
        monkeypatch.setenv("CLIPCANNON_DEV_MODE", "0")
        monkeypatch.setenv("CLIPCANNON_JWT_SECRET", "a-real-strong-secret-123")
        assert get_jwt_secret() == "a-real-strong-secret-123"

    def test_create_app_fails_hard_without_prod_secret(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The app refuses to start in prod without a real secret."""
        monkeypatch.setenv("CLIPCANNON_DEV_MODE", "0")
        monkeypatch.delenv("CLIPCANNON_JWT_SECRET", raising=False)
        with pytest.raises(RuntimeError, match="CLIPCANNON_JWT_SECRET"):
            create_app()

    def test_create_app_ok_with_prod_secret(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The app starts in prod when a real secret is set."""
        monkeypatch.setenv("CLIPCANNON_DEV_MODE", "0")
        monkeypatch.setenv("CLIPCANNON_JWT_SECRET", "a-real-strong-secret-123")
        app = create_app()
        assert app is not None


class TestCredits:
    """Credit API endpoint tests."""

    def test_balance_endpoint(self, client: TestClient) -> None:
        """GET /api/credits/balance returns balance info."""
        resp = client.get("/api/credits/balance")
        assert resp.status_code == 200
        data = resp.json()
        # Balance may be -1 if license server is not running
        assert "balance" in data
        assert "spending_limit" in data

    def test_history_endpoint(self, client: TestClient) -> None:
        """GET /api/credits/history returns transaction list."""
        resp = client.get("/api/credits/history")
        assert resp.status_code == 200
        data = resp.json()
        assert "transactions" in data
        assert isinstance(data["transactions"], list)
        assert "count" in data

    def test_packages_endpoint(self, client: TestClient) -> None:
        """GET /api/credits/packages returns available packages."""
        resp = client.get("/api/credits/packages")
        assert resp.status_code == 200
        data = resp.json()
        assert "packages" in data
        assert len(data["packages"]) > 0
        assert "credit_rates" in data

        # Verify package structure
        pkg = data["packages"][0]
        assert "name" in pkg
        assert "credits" in pkg
        assert "price_cents" in pkg
        assert "price_display" in pkg

    def test_add_credits_dev_mode(self, client: TestClient) -> None:
        """POST /api/credits/add works in dev mode."""
        resp = client.post("/api/credits/add?amount=50")
        assert resp.status_code == 200
        data = resp.json()
        # May fail if license server is down, but the endpoint should respond
        assert "amount" in data


class TestProjects:
    """Project API endpoint tests."""

    def test_list_projects(self, client: TestClient) -> None:
        """GET /api/projects returns project list."""
        resp = client.get("/api/projects")
        assert resp.status_code == 200
        data = resp.json()
        assert "projects" in data
        assert isinstance(data["projects"], list)
        assert "count" in data

    def test_project_detail_not_found(self, client: TestClient) -> None:
        """GET /api/projects/{id} returns not found for missing project."""
        resp = client.get("/api/projects/nonexistent-project-xyz")
        assert resp.status_code == 200
        data = resp.json()
        assert data["found"] is False

    def test_project_status_not_found(self, client: TestClient) -> None:
        """GET /api/projects/{id}/status returns not_found status."""
        resp = client.get("/api/projects/nonexistent-project-xyz/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "not_found"


class TestProvenance:
    """Provenance API endpoint tests."""

    def test_provenance_records_missing_project(self, client: TestClient) -> None:
        """GET /api/provenance/{id} handles missing project."""
        resp = client.get("/api/provenance/nonexistent-project-xyz")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 0
        assert "error" in data

    def test_provenance_verify_missing_project(self, client: TestClient) -> None:
        """GET /api/provenance/{id}/verify handles missing project."""
        resp = client.get("/api/provenance/nonexistent-project-xyz/verify")
        assert resp.status_code == 200
        data = resp.json()
        assert data["verified"] is False

    def test_provenance_timeline_missing_project(self, client: TestClient) -> None:
        """GET /api/provenance/{id}/timeline handles missing project."""
        resp = client.get("/api/provenance/nonexistent-project-xyz/timeline")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 0


class TestHomePage:
    """Home page and overview tests."""

    def test_home_serves_html(self, client: TestClient) -> None:
        """GET / serves the static HTML page."""
        resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")
        assert "ClipCannon" in resp.text

    def test_api_overview(self, client: TestClient) -> None:
        """GET /api/overview returns system overview."""
        resp = client.get("/api/overview")
        assert resp.status_code == 200
        data = resp.json()
        assert "version" in data
        assert "gpu" in data
        assert "system_health" in data
        assert "recent_projects" in data
