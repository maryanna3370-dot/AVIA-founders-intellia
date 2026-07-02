import unittest


class TestPlannerExecutor(unittest.TestCase):
    def test_basic_flow(self):
        from agent.planner import Planner
        from agent.executor import Executor

        p = Planner()
        plan = p.decompose("Get examples of product-market fit.")
        self.assertTrue(len(plan) >= 1)

        exe = Executor()
        res = exe.execute_step(plan[0])
        self.assertEqual(res.get("status"), "ok")
        self.assertIn("summary", res.get("output"))


if __name__ == "__main__":
    unittest.main()
