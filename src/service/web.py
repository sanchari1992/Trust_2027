"""The web layer: a small FastAPI app. GET / = the page, GET /api/mutations = what can be chosen,
POST /api/mutate = do it. All logic lives in src/service/mutator.py."""
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from src.service.mutator import MutationService, RequestError

WEB_DIR = Path(__file__).resolve().parent.parent.parent / "web"


class MutateRequest(BaseModel):
    question: str = Field(..., description="The prompt / question to mutate.")
    answer: str = Field("", description="Its expected answer (needed for Premise).")
    evidence: str = Field("", description="Optional: the source text the answer comes from.")
    mutations: List[str] = Field(["elaborate", "colloquial", "premise"],
                                 description="Mutation names (elaborate, colloquial, premise) and/or operator names (typos, ...).")
    composite: bool = Field(False, description="Also return all chosen mutations applied together in one prompt.")
    premise_kind: str = Field("either", description="What Premise adds: either | correct | incorrect (a statement by the user) | patient_fits | patient_conflicts | patient_either (a patient profile).")
    patient_profile: str = Field("", description="Optional: your own patient profile text, e.g. 'I am a 7-year-old girl.' It is used as the Premise.")
    profile_relation: str = Field("fits", description="For patient_profile: 'fits' (answer unchanged) or 'conflicts' (expected answer: Cannot be answered from given information.).")
    depth: Optional[int] = Field(None, description="How many operators to stack (1-4); empty = default.")
    seed: Optional[int] = Field(None, description="Fixed seed for a repeatable result.")
    randomize: bool = Field(True, description="true = a different result each time; false = repeatable.")


def create_app(cfg, token=None, chat=None):
    app = FastAPI(title="Benchmark+ mutator", description="Mutate one prompt and get the new expected answer.")
    service = MutationService(cfg, chat=chat)

    @app.get("/")
    def page():
        return FileResponse(WEB_DIR / "index.html")

    @app.get("/api/mutations")
    def mutations():
        return {"mutations": service.catalog(), "needs_token": bool(token)}

    @app.post("/api/mutate")
    def mutate(req: MutateRequest, x_token: Optional[str] = Header(None)):
        if token and x_token != token:
            raise HTTPException(401, "Wrong or missing token.")
        try:
            return service.mutate(req.question, req.answer, req.evidence, req.mutations, req.composite,
                                  req.premise_kind, req.depth, req.seed, req.randomize,
                                  req.patient_profile, req.profile_relation)
        except RequestError as e:
            raise HTTPException(400, str(e))
        except Exception as e:  # LLM key missing, network, ...
            msg = f"{type(e).__name__}: {e}"[:300]
            hint = " Check the DeepSeek key in .env and your balance." if "api" in msg.lower() or "auth" in msg.lower() or "402" in msg else ""
            return JSONResponse({"detail": msg + hint}, status_code=502)

    return app
