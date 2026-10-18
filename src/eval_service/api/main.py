"""FastAPI surface. Runs execute in-process for now; Day 5 swaps in the queue.

    uvicorn eval_service.api.main:app --app-dir src

POST /evals            submit an EvalJob -> returns run_id
GET  /evals            list past runs (model, task, prompt_version, metrics)
GET  /evals/{run_id}   one run incl. cost/latency
GET  /compare?a=&b=    regression diff between two runs
"""
from __future__ import annotations

from fastapi import BackgroundTasks, FastAPI, HTTPException

from ..runner.executor import run_job
from ..runner.jobs import EvalJob
from ..storage.db import Store

app = FastAPI(title="LLM Eval Service", version="0.1.0")
store = Store()


@app.post("/evals")
def submit_eval(job: EvalJob, background_tasks: BackgroundTasks):
    job.id = job.id or __import__("uuid").uuid4().hex[:12]
    background_tasks.add_task(run_job, job, store)  # TODO(Day 5): queue.submit(job)
    return {"run_id": job.id, "status": job.status}


@app.get("/evals")
def list_evals():
    return store.list_runs()


@app.get("/evals/{run_id}")
def get_eval(run_id: str):
    run = store.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="unknown run id")
    return run


@app.get("/compare")
def compare(a: str, b: str):
    try:
        return store.compare_runs(a, b)
    except KeyError:
        raise HTTPException(status_code=404, detail="unknown run id")
