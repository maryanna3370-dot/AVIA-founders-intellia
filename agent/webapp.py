"""Small FastAPI web UI for enqueueing tasks and viewing status."""
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from .scheduler import Scheduler
from .memory import get_default_memory
from fastapi import HTTPException

app = FastAPI()
_SCHED = Scheduler()
_MEM = get_default_memory()


@app.post("/enqueue")
def enqueue(task: str = Form(...)):
    tid = _SCHED.enqueue(task)
    return {"task_id": tid}


@app.get("/task/{task_id}")
def task_status(task_id: str):
    steps = _MEM.get_task_steps(task_id)
    return {"task_id": task_id, "steps": steps}

@app.get("/pending")
def pending_steps():
    steps = _MEM.get_pending_steps()
    return {"count": len(steps), "pending": steps}


@app.post("/approve")
def approve(task_id: str = Form(...), step_index: int = Form(...)):
    # approve a step so scheduler workers can proceed
    try:
        _MEM.approve_step(task_id, int(step_index))
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(
        """
        <html><body>
        <h3>Agent UI</h3>
        <form action="/enqueue" method="post">
          <textarea name="task" rows="4" cols="60">Summarize market sizing.</textarea><br/>
          <button type="submit">Enqueue</button>
        </form>
        <p>Use /task/{id} to inspect results.</p>
                <p><a href="/dashboard">Approvals Dashboard</a></p>
                </body></html>
        """
    )


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
        # simple approvals dashboard
        return HTMLResponse(
                """
                <html>
                <head>
                <title>Approvals Dashboard</title>
                </head>
                <body>
                <h2>Pending Steps</h2>
                <div id="pending"></div>
                <script>
                async function loadPending(){
                    const r = await fetch('/pending');
                    const j = await r.json();
                    const container = document.getElementById('pending');
                    container.innerHTML = '';
                    if(j.count === 0){ container.innerHTML = '<p>No pending steps</p>'; return }
                    j.pending.forEach(p => {
                        const d = document.createElement('div');
                        d.style.border='1px solid #ddd'; d.style.padding='8px'; d.style.margin='6px';
                        d.innerHTML = `<b>Task:</b> ${p.task_id} <b>Step:</b> ${p.step_index} <div>${p.step.action}</div>`;
                        const btn = document.createElement('button'); btn.textContent = 'Approve';
                        btn.onclick = async ()=>{
                            const fd = new FormData(); fd.append('task_id', p.task_id); fd.append('step_index', p.step_index);
                            await fetch('/approve', {method:'POST', body: fd}); loadPending();
                        };
                        d.appendChild(btn);
                        container.appendChild(d);
                    });
                }
                loadPending(); setInterval(loadPending, 5000);
                </script>
                </body>
                </html>
                """
        )
