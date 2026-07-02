import unittest
import time


class TestMemoryScheduler(unittest.TestCase):
    def test_scheduler_runs_and_records(self):
        from agent.scheduler import Scheduler
        from agent.memory import Memory

        s = Scheduler(worker_count=1)
        tid = s.enqueue("Find competitors and summarize market sizing.")
        # wait for work to be processed
        time.sleep(0.8)
        steps = s.memory.get_task_steps(tid)
        s.stop()
        self.assertTrue(len(steps) >= 1)
        self.assertIn("result", steps[0])


if __name__ == "__main__":
    unittest.main()
