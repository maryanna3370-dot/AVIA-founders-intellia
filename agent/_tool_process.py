"""Helper module executed as a subprocess to run a tool safely.

Usage: python -m agent._tool_process <tool_module>
Reads a JSON object from STDIN with `args`, imports the tool module and calls `run(args)`
and writes JSON to STDOUT.
"""
import sys
import json

def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "missing_module"}))
        sys.exit(2)
    module_name = sys.argv[1]
    data = json.load(sys.stdin)
    args = data.get("args", {})
    try:
        mod = __import__(module_name, fromlist=["*"])
        if not hasattr(mod, "run"):
            print(json.dumps({"error": "no_run"}))
            sys.exit(2)
        out = mod.run(args)
        print(json.dumps({"ok": True, "out": out}))
    except Exception as e:
        print(json.dumps({"ok": False, "err": str(e)}))
        sys.exit(1)

if __name__ == "__main__":
    main()
