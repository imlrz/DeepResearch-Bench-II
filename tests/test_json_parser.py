import json
import sys
import types
import unittest


fake_gpt_client = types.ModuleType("gpt_client")
fake_gpt_client.GPTClient = object
fake_gpt_client.GPTInput = object
fake_gpt_client.get_config = lambda key, default: default
sys.modules.setdefault("gpt_client", fake_gpt_client)

from run_evaluation import parse_model_text


class ParseModelTextTests(unittest.TestCase):
    def test_parses_valid_json_without_rewriting_it(self):
        expected = {
            "results": [
                {
                    "rubric_item": "a",
                    "score": 1,
                    "reason": "ok",
                    "evidence": "x",
                }
            ]
        }

        parsed, ok = parse_model_text(json.dumps(expected))

        self.assertTrue(ok)
        self.assertEqual(parsed, expected)

    def test_parses_valid_json_from_a_fenced_block(self):
        expected = {
            "results": [
                {
                    "rubric_item": "a",
                    "score": 1,
                    "reason": "ok",
                    "evidence": "x",
                }
            ]
        }
        text = f"```json\n{json.dumps(expected)}\n```"

        parsed, ok = parse_model_text(text)

        self.assertTrue(ok)
        self.assertEqual(parsed, expected)

    def test_keeps_malformed_key_cleanup_as_a_fallback(self):
        parsed, ok = parse_model_text('{"rubric_"item": 1}')

        self.assertTrue(ok)
        self.assertEqual(parsed, {'rubric_"item': 1})
