"""Tests for SessionResolver — session identity resolution."""

import uuid

from capture_proxy.session import SessionResolver


class TestResolve:
    """Tests for SessionResolver.resolve()."""

    def setup_method(self) -> None:
        self.resolver = SessionResolver()

    def test_x_session_id_used(self) -> None:
        """x-session-id header is returned as session_id."""
        headers = {"x-session-Id": "my-session-123"}
        result = self.resolver.resolve(headers)
        assert result == "my-session-123"

    def test_x_session_id_case_insensitive(self) -> None:
        """x-session-id matched case-insensitively."""
        headers = {"X-SESSION-ID": "session-abc"}
        result = self.resolver.resolve(headers)
        assert result == "session-abc"

    def test_auth_header_hashed_to_uuid(self) -> None:
        """Authorization header hashed to deterministic UUID v4."""
        headers = {"Authorization": "Bearer sk-secret-key"}
        result = self.resolver.resolve(headers)
        # Should be valid UUID v4
        parsed = uuid.UUID(result)
        assert parsed.version == 4

    def test_auth_deterministic(self) -> None:
        """Same Authorization header → same session_id."""
        headers = {"Authorization": "Bearer sk-secret-key"}
        r1 = self.resolver.resolve(headers)
        r2 = self.resolver.resolve(headers)
        assert r1 == r2

    def test_different_auth_different_uuids(self) -> None:
        """Different Authorization headers → different session_ids."""
        h1 = {"Authorization": "Bearer key-one"}
        h2 = {"Authorization": "Bearer key-two"}
        assert self.resolver.resolve(h1) != self.resolver.resolve(h2)

    def test_neither_header_random_uuid(self) -> None:
        """No x-session-id or Authorization → random UUID v4."""
        result = self.resolver.resolve({})
        parsed = uuid.UUID(result)
        assert parsed.version == 4

    def test_random_uuid_unique(self) -> None:
        """Each call without headers produces different UUID."""
        r1 = self.resolver.resolve({})
        r2 = self.resolver.resolve({})
        assert r1 != r2

    def test_x_session_id_takes_priority(self) -> None:
        """x-session-id takes priority over Authorization."""
        headers = {
            "x-session-id": "explicit-session",
            "Authorization": "Bearer sk-key",
        }
        result = self.resolver.resolve(headers)
        assert result == "explicit-session"

    def test_empty_headers_returns_uuid(self) -> None:
        """Empty headers dict returns valid UUID."""
        result = self.resolver.resolve({})
        parsed = uuid.UUID(result)
        assert parsed.version == 4

    def test_returns_string(self) -> None:
        """Result is always a string."""
        result = self.resolver.resolve({})
        assert isinstance(result, str)
