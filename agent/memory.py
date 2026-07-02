"""Lightweight persistent memory using SQLite.

Purpose: store tasks, steps, and observations for provenance and simple queries.
This keeps dependencies minimal and works offline.
"""
import sqlite3
import json
from typing import Optional, Dict, Any, List
import os
try:
    import redis
except Exception:
    redis = None
try:
    import psycopg2
except Exception:
    psycopg2 = None


DB_PATH = os.environ.get("AGENT_MEMORY_DB", "./agent_memory.db")


class Memory:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._init_db()

    def _init_db(self):
        c = self._conn.cursor()
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                prompt TEXT,
                meta TEXT
            )
            """
        )
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS steps (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT,
                step_index INTEGER,
                step_json TEXT,
                result_json TEXT
            )
            """
        )
        # Migrate: ensure `status` and `approved` columns exist
        cols = [r[1] for r in c.execute("PRAGMA table_info(steps)")]
        if "status" not in cols:
            try:
                c.execute("ALTER TABLE steps ADD COLUMN status TEXT DEFAULT 'pending'")
            except Exception:
                pass
        if "approved" not in cols:
            try:
                c.execute("ALTER TABLE steps ADD COLUMN approved INTEGER DEFAULT 0")
            except Exception:
                pass
        self._conn.commit()

    def save_task(self, task_id: str, prompt: str, meta: Optional[Dict[str, Any]] = None):
        c = self._conn.cursor()
        c.execute("REPLACE INTO tasks(id,prompt,meta) VALUES(?,?,?)", (task_id, prompt, json.dumps(meta or {})))
        self._conn.commit()

    def save_step_result(self, task_id: str, step_index: int, step: Dict[str, Any], result: Dict[str, Any]):
        c = self._conn.cursor()
        c.execute(
            "INSERT INTO steps(task_id,step_index,step_json,result_json,status,approved) VALUES(?,?,?,?,?,?)",
            (task_id, step_index, json.dumps(step), json.dumps(result), "done", 1),
        )
        self._conn.commit()

    def save_step_pending(self, task_id: str, step_index: int, step: Dict[str, Any]):
        c = self._conn.cursor()
        c.execute(
            "INSERT INTO steps(task_id,step_index,step_json,status,approved) VALUES(?,?,?,?,?)",
            (task_id, step_index, json.dumps(step), "pending", 0),
        )
        self._conn.commit()

    def approve_step(self, task_id: str, step_index: int):
        c = self._conn.cursor()
        c.execute("UPDATE steps SET approved=1 WHERE task_id=? AND step_index=?", (task_id, step_index))
        self._conn.commit()

    def get_pending_steps(self) -> List[Dict[str, Any]]:
        c = self._conn.cursor()
        c.execute("SELECT task_id, step_index, step_json FROM steps WHERE status='pending' AND approved=0 ORDER BY rowid")
        rows = c.fetchall()
        out = []
        for task_id, idx, sjson in rows:
            out.append({"task_id": task_id, "step_index": idx, "step": json.loads(sjson)})
        return out

    def get_task_steps(self, task_id: str) -> List[Dict[str, Any]]:
        c = self._conn.cursor()
        c.execute("SELECT step_index, step_json, result_json FROM steps WHERE task_id=? ORDER BY step_index", (task_id,))
        rows = c.fetchall()
        out = []
        for idx, sjson, rjson in rows:
            step = json.loads(sjson) if sjson else None
            result = json.loads(rjson) if rjson else None
            out.append({"step_index": idx, "step": step, "result": result})
        return out


_DEFAULT_MEMORY = None


def get_default_memory() -> Memory:
    global _DEFAULT_MEMORY
    if _DEFAULT_MEMORY is None:
        # Use Redis if configured
        if os.environ.get("REDIS_URL") and redis is not None:
            _DEFAULT_MEMORY = RedisMemory(os.environ.get("REDIS_URL"))
        # Use Postgres if configured
        elif os.environ.get("POSTGRES_DSN") and psycopg2 is not None:
            _DEFAULT_MEMORY = PostgresMemory(os.environ.get("POSTGRES_DSN"))
        else:
            _DEFAULT_MEMORY = Memory()
    return _DEFAULT_MEMORY



class RedisMemory(Memory):
    def __init__(self, url: str):
        # lightweight wrapper storing JSON strings
        self._r = redis.from_url(url)

    def save_task(self, task_id: str, prompt: str, meta: Optional[Dict[str, Any]] = None):
        self._r.hset(f"task:{task_id}", mapping={"prompt": prompt, "meta": json.dumps(meta or {})})

    def save_step_result(self, task_id: str, step_index: int, step: Dict[str, Any], result: Dict[str, Any]):
        self._r.rpush(f"steps:{task_id}", json.dumps({"step_index": step_index, "step": step, "result": result}))

    def get_task_steps(self, task_id: str) -> List[Dict[str, Any]]:
        items = self._r.lrange(f"steps:{task_id}", 0, -1)
        out = [json.loads(i) for i in items]
        return out

    def wait_for_approval(self, task_id: str, step_index: int, timeout: float) -> bool:
        # subscribe to approval channel and wait until approval message for this task/step arrives
        pubsub = self._r.pubsub()
        channel = f"approval:{task_id}:{step_index}"
        pubsub.subscribe(channel)
        start = time.time()
        for msg in pubsub.listen():
            if msg is None:
                continue
            if msg.get("type") == "message":
                try:
                    data = json.loads(msg.get("data") or b"{}")
                except Exception:
                    data = {}
                if data.get("task_id") == task_id and int(data.get("step_index", -1)) == int(step_index):
                    pubsub.close()
                    return True
            if time.time() - start > timeout:
                pubsub.close()
                return False

    def approve_step(self, task_id: str, step_index: int):
        super().approve_step(task_id, step_index)
        # publish approval event
        channel = f"approval:{task_id}:{step_index}"
        self._r.publish(channel, json.dumps({"task_id": task_id, "step_index": step_index}))


class PostgresMemory(Memory):
    def __init__(self, dsn: str):
        if psycopg2 is None:
            raise RuntimeError("psycopg2 not installed")
        self._conn = psycopg2.connect(dsn)

    def save_task(self, task_id: str, prompt: str, meta: Optional[Dict[str, Any]] = None):
        with self._conn.cursor() as c:
            c.execute("CREATE TABLE IF NOT EXISTS tasks_pg (id TEXT PRIMARY KEY, prompt TEXT, meta JSONB)")
            c.execute("INSERT INTO tasks_pg (id,prompt,meta) VALUES (%s,%s,%s) ON CONFLICT (id) DO UPDATE SET prompt=EXCLUDED.prompt", (task_id, prompt, json.dumps(meta or {})))
            self._conn.commit()

    def save_step_result(self, task_id: str, step_index: int, step: Dict[str, Any], result: Dict[str, Any]):
        with self._conn.cursor() as c:
            c.execute("CREATE TABLE IF NOT EXISTS steps_pg (id SERIAL PRIMARY KEY, task_id TEXT, step_index INT, step_json JSONB, result_json JSONB)")
            c.execute("INSERT INTO steps_pg (task_id,step_index,step_json,result_json) VALUES (%s,%s,%s,%s)", (task_id, step_index, json.dumps(step), json.dumps(result)))
            self._conn.commit()

    def get_task_steps(self, task_id: str) -> List[Dict[str, Any]]:
        with self._conn.cursor() as c:
            c.execute("SELECT step_index, step_json, result_json FROM steps_pg WHERE task_id=%s ORDER BY step_index", (task_id,))
            rows = c.fetchall()
            out = []
            for idx, sjson, rjson in rows:
                out.append({"step_index": idx, "step": sjson, "result": rjson})
            return out


if __name__ == "__main__":
    m = Memory()
    m.save_task("t1", "Example prompt")
    m.save_step_result("t1", 1, {"action":"x"}, {"status":"ok"})
    print(m.get_task_steps("t1"))
