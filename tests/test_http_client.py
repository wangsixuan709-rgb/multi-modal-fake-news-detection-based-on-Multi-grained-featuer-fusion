import unittest
from unittest.mock import MagicMock, patch

import requests

from blockchain_bridge import (
    PredictionPayload,
    health_check,
    submit_proof,
    submit_proof_with_retry,
    verify_audit,
)


def _make_payload(**kwargs):
    defaults = dict(
        dataset="weibo",
        image_path="/data/img.jpg",
        predicted_label=1,
        confidence=0.9,
    )
    defaults.update(kwargs)
    return PredictionPayload(**defaults)


class HealthCheckTests(unittest.TestCase):
    @patch("blockchain_bridge.requests.get")
    def test_returns_true_on_200(self, mock_get):
        mock_get.return_value = MagicMock(status_code=200)
        self.assertTrue(health_check("http://localhost:3000"))

    @patch("blockchain_bridge.requests.get")
    def test_returns_true_on_any_http_response(self, mock_get):
        mock_get.return_value = MagicMock(status_code=404)
        self.assertTrue(health_check("http://localhost:3000"))

    @patch("blockchain_bridge.requests.get", side_effect=requests.ConnectionError)
    def test_returns_false_on_connection_error(self, _mock_get):
        self.assertFalse(health_check("http://localhost:3000"))

    @patch("blockchain_bridge.requests.get", side_effect=requests.Timeout)
    def test_returns_false_on_timeout(self, _mock_get):
        self.assertFalse(health_check("http://localhost:3000"))


class SubmitProofTests(unittest.TestCase):
    @patch("blockchain_bridge.requests.post")
    def test_returns_json_on_success(self, mock_post):
        mock_post.return_value = MagicMock(
            status_code=201,
            json=lambda: {"receipt_id": "abc123"},
        )
        mock_post.return_value.raise_for_status = MagicMock()

        result = submit_proof(_make_payload(), base_url="http://localhost:3000")
        self.assertEqual(result, {"receipt_id": "abc123"})
        mock_post.assert_called_once()
        self.assertIn("/prove", mock_post.call_args.args[0])

    @patch("blockchain_bridge.requests.post")
    def test_raises_on_http_error(self, mock_post):
        mock_response = MagicMock(status_code=500)
        mock_response.raise_for_status.side_effect = requests.HTTPError("500 Server Error")
        mock_post.return_value = mock_response

        with self.assertRaises(requests.HTTPError):
            submit_proof(_make_payload(), base_url="http://localhost:3000")


class SubmitProofWithRetryTests(unittest.TestCase):
    @patch("blockchain_bridge.requests.post")
    def test_succeeds_on_first_attempt(self, mock_post):
        mock_post.return_value = MagicMock(
            json=lambda: {"receipt_id": "xyz"},
        )
        mock_post.return_value.raise_for_status = MagicMock()

        result = submit_proof_with_retry(_make_payload(), max_retries=3, base_url="http://localhost:3000")
        self.assertEqual(result["receipt_id"], "xyz")
        self.assertEqual(mock_post.call_count, 1)

    @patch("blockchain_bridge.time.sleep")
    @patch("blockchain_bridge.requests.post")
    def test_retries_on_connection_error(self, mock_post, mock_sleep):
        mock_post.side_effect = [
            requests.ConnectionError("refused"),
            requests.ConnectionError("refused"),
            MagicMock(json=lambda: {"receipt_id": "retry_ok"}, raise_for_status=MagicMock()),
        ]

        result = submit_proof_with_retry(_make_payload(), max_retries=3, backoff_seconds=1.0, base_url="http://localhost:3000")
        self.assertEqual(result["receipt_id"], "retry_ok")
        self.assertEqual(mock_post.call_count, 3)
        self.assertEqual(mock_sleep.call_count, 2)

    @patch("blockchain_bridge.time.sleep")
    @patch("blockchain_bridge.requests.post")
    def test_raises_after_max_retries_exhausted(self, mock_post, mock_sleep):
        mock_post.side_effect = requests.ConnectionError("refused")

        with self.assertRaises(requests.ConnectionError):
            submit_proof_with_retry(_make_payload(), max_retries=3, backoff_seconds=0.0, base_url="http://localhost:3000")

        self.assertEqual(mock_post.call_count, 3)

    def test_raises_value_error_when_max_retries_is_zero(self):
        with self.assertRaises(ValueError):
            submit_proof_with_retry(_make_payload(), max_retries=0, base_url="http://localhost:3000")

    @patch("blockchain_bridge.time.sleep")
    @patch("blockchain_bridge.requests.post")
    def test_exponential_backoff_timing(self, mock_post, mock_sleep):
        mock_post.side_effect = [
            requests.ConnectionError(),
            requests.ConnectionError(),
            MagicMock(json=lambda: {}, raise_for_status=MagicMock()),
        ]

        submit_proof_with_retry(_make_payload(), max_retries=3, backoff_seconds=2.0, base_url="http://localhost:3000")
        # First retry: 2.0 * 2^0 = 2.0; second retry: 2.0 * 2^1 = 4.0
        sleep_calls = [call[0][0] for call in mock_sleep.call_args_list]
        self.assertEqual(sleep_calls, [2.0, 4.0])


class VerifyAuditTests(unittest.TestCase):
    @patch("blockchain_bridge.requests.get")
    def test_returns_verification_result(self, mock_get):
        mock_get.return_value = MagicMock(
            json=lambda: {"receipt_id": "abc123", "valid": True},
        )
        mock_get.return_value.raise_for_status = MagicMock()

        result = verify_audit("abc123", base_url="http://localhost:3000")
        self.assertEqual(result["valid"], True)
        call_url = mock_get.call_args[0][0]
        self.assertIn("abc123", call_url)
        self.assertIn("/audit/", call_url)

    @patch("blockchain_bridge.requests.get")
    def test_raises_on_not_found(self, mock_get):
        mock_response = MagicMock(status_code=404)
        mock_response.raise_for_status.side_effect = requests.HTTPError("404 Not Found")
        mock_get.return_value = mock_response

        with self.assertRaises(requests.HTTPError):
            verify_audit("nonexistent", base_url="http://localhost:3000")
