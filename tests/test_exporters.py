# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
#
# Modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

"""Tests for opensearch_genai_observability_sdk_py.exporters.

Covers _SigV4AuthSession (header injection and body-hash correctness)
and AWSSigV4OTLPExporter (initialization guards).
No real AWS credentials or network calls are made."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import botocore.auth
import pytest
from botocore.awsrequest import AWSRequest
from botocore.credentials import Credentials

from opensearch_genai_observability_sdk_py.exporters import AWSSigV4OTLPExporter, _SigV4AuthSession

# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------

ENDPOINT = "https://pipeline.us-east-1.osis.amazonaws.com/v1/traces"
REGION = "us-east-1"
SERVICE = "osis"

# Static credentials — never touch real AWS.
FAKE_CREDS = Credentials(
    access_key="AKIAIOSFODNN7EXAMPLE",
    secret_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",  # noqa: S106
    token="FakeSessionToken",  # noqa: S106
)

FAKE_CREDS_NO_TOKEN = Credentials(
    access_key="AKIAIOSFODNN7EXAMPLE",
    secret_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",  # noqa: S106
)


def _make_session(credentials=FAKE_CREDS) -> _SigV4AuthSession:
    return _SigV4AuthSession(credentials, service=SERVICE, region=REGION)


# ---------------------------------------------------------------------------
# _SigV4AuthSession — header injection
# ---------------------------------------------------------------------------


class TestSigV4AuthSessionHeaders:
    """Verify that SigV4 auth headers are injected into outgoing requests."""

    @patch("requests.Session.request")
    def test_authorization_header_present(self, mock_request):
        mock_request.return_value = MagicMock(status_code=200)
        _make_session().request("POST", ENDPOINT, data=b"some-proto-bytes")

        headers = mock_request.call_args.kwargs["headers"]
        assert "Authorization" in headers

    @patch("requests.Session.request")
    def test_x_amz_date_header_present(self, mock_request):
        mock_request.return_value = MagicMock(status_code=200)
        _make_session().request("POST", ENDPOINT, data=b"payload")

        headers = mock_request.call_args.kwargs["headers"]
        assert "X-Amz-Date" in headers

    @patch("requests.Session.request")
    def test_security_token_injected_for_temp_credentials(self, mock_request):
        """X-Amz-Security-Token must be present when using temporary credentials."""
        mock_request.return_value = MagicMock(status_code=200)
        _make_session(FAKE_CREDS).request("POST", ENDPOINT, data=b"payload")

        headers = mock_request.call_args.kwargs["headers"]
        assert "X-Amz-Security-Token" in headers
        assert headers["X-Amz-Security-Token"] == "FakeSessionToken"

    @patch("requests.Session.request")
    def test_no_security_token_for_long_term_credentials(self, mock_request):
        """X-Amz-Security-Token must not appear for long-term (non-STS) credentials."""
        mock_request.return_value = MagicMock(status_code=200)
        _make_session(FAKE_CREDS_NO_TOKEN).request("POST", ENDPOINT, data=b"payload")

        headers = mock_request.call_args.kwargs["headers"]
        assert "X-Amz-Security-Token" not in headers

    @patch("requests.Session.request")
    def test_existing_headers_are_preserved(self, mock_request):
        """Headers passed by the caller (e.g. Content-Type) survive signing."""
        mock_request.return_value = MagicMock(status_code=200)
        incoming_headers = {"Content-Type": "application/x-protobuf", "X-Custom": "value"}
        _make_session().request("POST", ENDPOINT, data=b"payload", headers=incoming_headers)

        headers = mock_request.call_args.kwargs["headers"]
        assert headers.get("X-Custom") == "value"
        # SigV4 headers are added on top
        assert "Authorization" in headers

    @patch("requests.Session.request")
    def test_none_headers_treated_as_empty(self, mock_request):
        """Passing headers=None should not crash — auth headers are still injected."""
        mock_request.return_value = MagicMock(status_code=200)
        _make_session().request("POST", ENDPOINT, data=b"payload", headers=None)

        headers = mock_request.call_args.kwargs["headers"]
        assert "Authorization" in headers

    @patch("requests.Session.request")
    def test_real_data_forwarded_to_parent(self, mock_request):
        """The real payload must be forwarded to requests.Session.request unchanged."""
        mock_request.return_value = MagicMock(status_code=200)
        payload = b"this-is-real-protobuf"
        _make_session().request("POST", ENDPOINT, data=payload)

        assert mock_request.call_args.kwargs["data"] == payload


# ---------------------------------------------------------------------------
# _SigV4AuthSession — body hash correctness (regression for empty-body bug)
# ---------------------------------------------------------------------------


class TestSigV4BodyHash:
    """
    Verify that the signature is computed over the *real* request body,
    not a placeholder empty string.

    AWS SigV4 includes SHA256(body) in the canonical request.  If the
    SDK signed over b"" while sending a non-empty protobuf payload, AWS
    would compute a different body hash and return 403 SignatureDoesNotMatch.

    These tests confirm that different bodies produce different signatures,
    which is only possible if the body is actually hashed.
    """

    def _capture_auth_header(self, mock_request, data: bytes) -> str:
        mock_request.reset_mock()
        mock_request.return_value = MagicMock(status_code=200)
        _make_session().request("POST", ENDPOINT, data=data)
        return mock_request.call_args.kwargs["headers"]["Authorization"]

    @patch("requests.Session.request")
    def test_different_bodies_produce_different_signatures(self, mock_request):
        """Core regression: body hash must reflect the real payload."""
        sig_a = self._capture_auth_header(mock_request, b"body-variant-alpha")
        sig_b = self._capture_auth_header(mock_request, b"body-variant-beta")
        assert sig_a != sig_b, "Different bodies must produce different SigV4 signatures"

    @patch("requests.Session.request")
    def test_empty_body_signature_differs_from_real_body(self, mock_request):
        """
        Regression test for the original bug.

        The old implementation signed over data=b"" unconditionally, so
        the Authorization header always contained the empty-body hash
        regardless of the actual OTLP payload.  This test fails with the
        old approach and passes with the fixed one.
        """
        sig_empty = self._capture_auth_header(mock_request, b"")
        sig_real = self._capture_auth_header(mock_request, b"real-otlp-protobuf-payload")
        assert sig_empty != sig_real, (
            "Signing over an empty body must produce a different signature than signing "
            "over the real protobuf body.  If they are equal the body hash is not being "
            "computed correctly (likely the old empty-body-hash bug)."
        )

    @patch("requests.Session.request")
    def test_none_data_treated_as_empty_body(self, mock_request):
        """data=None must not crash — treated the same as b''."""
        mock_request.return_value = MagicMock(status_code=200)
        _make_session().request("POST", ENDPOINT, data=None)
        headers = mock_request.call_args.kwargs["headers"]
        assert "Authorization" in headers


# ---------------------------------------------------------------------------
# AWSSigV4OTLPExporter — initialization guards
# ---------------------------------------------------------------------------


class TestAWSSigV4OTLPExporterInit:
    """Verify initialization errors and correct session wiring."""

    def _mock_botocore(self, credentials=FAKE_CREDS, region: str | None = "us-east-1"):
        """Return a context manager that patches botocore.session.get_session()."""
        mock_session = MagicMock()
        mock_session.get_credentials.return_value = credentials
        mock_session.get_config_variable.return_value = region

        return patch("botocore.session.get_session", return_value=mock_session)

    def test_raises_if_no_credentials(self):
        with self._mock_botocore(credentials=None):
            with pytest.raises(RuntimeError, match="No AWS credentials found"):
                AWSSigV4OTLPExporter(endpoint=ENDPOINT, region=REGION)

    def test_raises_if_no_region(self):
        with self._mock_botocore(region=None):
            with pytest.raises(RuntimeError, match="No AWS region found"):
                AWSSigV4OTLPExporter(endpoint=ENDPOINT)

    def test_explicit_region_overrides_botocore(self):
        """region= kwarg takes precedence over whatever botocore detects."""
        with self._mock_botocore(region="ap-southeast-1"):
            exporter = AWSSigV4OTLPExporter(endpoint=ENDPOINT, region="eu-west-1")
        assert exporter._session._region == "eu-west-1"

    def test_session_is_sigv4_auth_session(self):
        """The exporter's internal session must be our signing session."""
        with self._mock_botocore():
            exporter = AWSSigV4OTLPExporter(endpoint=ENDPOINT)
        assert isinstance(exporter._session, _SigV4AuthSession)

    def test_session_carries_correct_service(self):
        with self._mock_botocore():
            exporter = AWSSigV4OTLPExporter(endpoint=ENDPOINT, service="es")
        assert exporter._session._service == "es"

    def test_default_service_is_osis(self):
        with self._mock_botocore():
            exporter = AWSSigV4OTLPExporter(endpoint=ENDPOINT)
        assert exporter._session._service == "osis"


# ---------------------------------------------------------------------------
# Canonical request structure — independent mathematical verification
# ---------------------------------------------------------------------------


class TestSigV4CanonicalRequest:
    """Verify that our code passes the real body to botocore."""

    @patch("requests.Session.request")
    def test_add_auth_receives_real_body_not_placeholder(self, mock_request):
        """_SigV4AuthSession must pass the real payload to SigV4Auth.add_auth(), not b''."""
        mock_request.return_value = MagicMock(status_code=200)
        payload = b"protobuf-spans-for-real"
        captured: list[AWSRequest] = []

        original_add_auth = botocore.auth.SigV4Auth.add_auth

        def intercepting_add_auth(self, request):
            captured.append(request)
            return original_add_auth(self, request)

        with patch.object(botocore.auth.SigV4Auth, "add_auth", intercepting_add_auth):
            _make_session().request("POST", ENDPOINT, data=payload)

        assert len(captured) == 1
        assert captured[0].body == payload
