import unittest


class TestTools(unittest.TestCase):
    def test_fs_search_finds_planner(self):
        from tools import fs_search
        res = fs_search.run({"query": "Planner", "max_results": 5, "root": "."})
        self.assertEqual(res.get("status"), "ok")
        self.assertTrue(len(res.get("matches", [])) >= 1)

    def test_web_search_simulated(self):
        from tools import web_search
        res = web_search.run({"query": "market sizing"})
        self.assertEqual(res.get("status"), "ok")
        self.assertIn("results", res)

    def test_http_api_simulated_missing_url(self):
        from tools import http_api
        res = http_api.run({"method": "GET"})
        self.assertEqual(res.get("status"), "error")


if __name__ == "__main__":
    unittest.main()
