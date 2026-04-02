import unittest
from unittest.mock import patch, Mock

import requests

from blockchain_bridge import (
    label_to_verdict,
    normalize_confidence,
    PredictionPayload,
    submit_proof,
    verify_audit,
    health_check,
    submit_proof_with_retry,
)


class BlockchainBridgeTests(unittest.TestCase):
    def test_weibo_label_mapping(self):
        self.assertFalse(label_to_verdict("weibo", 0))
        self.assertTrue(label_to_verdict("weibo", 1))

    def test_gossip_label_mapping(self):
        self.assertTrue(label_to_verdict("gossip", 0))
        self.assertFalse(label_to_verdict("gossip", 1))

    def test_confidence_is_clipped(self):
        self.assertEqual(normalize_confidence(-0.1), 0.0)
        self.assertEqual(normalize_confidence(1.8), 1.0)

    def test_prediction_payload_maps_to_api_shape(self):
        payload = PredictionPayload(
            dataset="gossip",
            image_path="/data/news.png",
            predicted_label=1,
            confidence=1.2,
            source="mmfn-eval",
        )

        self.assertEqual(
            payload.to_api_payload(),
            {
                "image_path": "/data/news.png",
                "verdict": False,
                "confidence": 1.0,
                "source": "mmfn-eval",
                "prompt_pool_hash": "0" * 64,
            },
        )


class BlockchainClientTests(unittest.TestCase):
    """Tests for HTTP client functions using mocked requests."""

    def _make_payload(self) -> PredictionPayload:
        return PredictionPayload(
            dataset="weibo",
            image_path="/tmp/test.jpg",
            predicted_label=1,
            confidence=0.95,
        )

    # ------------------------------------------------------------------
    # submit_proof
    # ------------------------------------------------------------------

    @patch("blockchain_bridge.requests.post")
    def test_submit_proof_success(self, mock_post):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"receipt_id": "abc123", "status": "ok"}
        mock_post.return_value = mock_response

        result = submit_proof(self._make_payload())

        self.assertEqual(result["receipt_id"], "abc123")
        mock_post.assert_called_once()

    @patch("blockchain_bridge.requests.post")
    def test_submit_proof_raises_on_http_error(self, mock_post):
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.HTTPError("500 Server Error")
        mock_post.return_value = mock_response

        with self.assertRaises(requests.HTTPError):
            submit_proof(self._make_payload())

    @patch("blockchain_bridge.requests.post")
    def test_submit_proof_timeout(self, mock_post):
        mock_post.side_effect = requests.Timeout("Connection timed out")

        with self.assertRaises(requests.Timeout):
            submit_proof(self._make_payload())

    @patch("blockchain_bridge.requests.post")
    def test_submit_proof_passes_custom_timeout(self, mock_post):
        mock_response = Mock()
        mock_response.json.return_value = {}
        mock_post.return_value = mock_response

        submit_proof(self._make_payload(), timeout=30.0)

        _, kwargs = mock_post.call_args
        self.assertEqual(kwargs["timeout"], 30.0)

    # ------------------------------------------------------------------
    # verify_audit
    # ------------------------------------------------------------------

    @patch("blockchain_bridge.requests.get")
    def test_verify_audit_success(self, mock_get):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"verified": True, "receipt_id": "abc123"}
        mock_get.return_value = mock_response

        result = verify_audit("abc123")

        self.assertTrue(result["verified"])
        mock_get.assert_called_once()

    @patch("blockchain_bridge.requests.get")
    def test_verify_audit_raises_on_http_error(self, mock_get):
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.HTTPError("404 Not Found")
        mock_get.return_value = mock_response

        with self.assertRaises(requests.HTTPError):
            verify_audit("nonexistent")

    # ------------------------------------------------------------------
    # health_check
    # ------------------------------------------------------------------

    @patch("blockchain_bridge.requests.get")
    def test_health_check_success(self, mock_get):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        self.assertTrue(health_check())

    @patch("blockchain_bridge.requests.get")
    def test_health_check_non_200_still_reachable(self, mock_get):
        mock_response = Mock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        self.assertTrue(health_check())

    @patch("blockchain_bridge.requests.get")
    def test_health_check_connection_error(self, mock_get):
        mock_get.side_effect = requests.ConnectionError()

        self.assertFalse(health_check())

    @patch("blockchain_bridge.requests.get")
    def test_health_check_timeout(self, mock_get):
        mock_get.side_effect = requests.Timeout()

        self.assertFalse(health_check())

    # ------------------------------------------------------------------
    # submit_proof_with_retry
    # ------------------------------------------------------------------

    @patch("blockchain_bridge.requests.post")
    def test_submit_proof_with_retry_succeeds_on_first_attempt(self, mock_post):
        mock_response = Mock()
        mock_response.json.return_value = {"receipt_id": "xyz"}
        mock_post.return_value = mock_response

        result = submit_proof_with_retry(self._make_payload())

        self.assertEqual(result["receipt_id"], "xyz")
        self.assertEqual(mock_post.call_count, 1)

    @patch("blockchain_bridge.time.sleep", return_value=None)
    @patch("blockchain_bridge.requests.post")
    def test_submit_proof_with_retry_retries_on_timeout(self, mock_post, mock_sleep):
        mock_response = Mock()
        mock_response.json.return_value = {"receipt_id": "xyz"}
        mock_post.side_effect = [
            requests.Timeout("timeout"),
            mock_response,
        ]

        result = submit_proof_with_retry(self._make_payload(), max_retries=3, backoff_seconds=1.0)

        self.assertEqual(result["receipt_id"], "xyz")
        self.assertEqual(mock_post.call_count, 2)
        mock_sleep.assert_called_once_with(1.0)

    @patch("blockchain_bridge.time.sleep", return_value=None)
    @patch("blockchain_bridge.requests.post")
    def test_submit_proof_with_retry_raises_after_all_retries(self, mock_post, mock_sleep):
        mock_post.side_effect = requests.Timeout("persistent timeout")

        with self.assertRaises(requests.Timeout):
            submit_proof_with_retry(self._make_payload(), max_retries=3, backoff_seconds=1.0)

        self.assertEqual(mock_post.call_count, 3)
        self.assertEqual(mock_sleep.call_count, 2)

    @patch("blockchain_bridge.requests.post")
    def test_submit_proof_with_retry_does_not_retry_http_error(self, mock_post):
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.HTTPError("400 Bad Request")
        mock_post.return_value = mock_response

        with self.assertRaises(requests.HTTPError):
            submit_proof_with_retry(self._make_payload(), max_retries=3)

        self.assertEqual(mock_post.call_count, 1)
