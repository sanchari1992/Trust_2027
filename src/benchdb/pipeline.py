"""Where the databases of a pipeline run live, and how to (re)build them."""
from src.config import ROOT, corpus_tag


def db_dir(cfg):
    return ROOT / cfg["benchdb"]["dir"]


def real_db_path(cfg, domain):
    return db_dir(cfg) / f"real_{domain}.benchmark"


def run_db_path(cfg, rag):
    return db_dir(cfg) / f"{rag}_{corpus_tag(cfg, rag)}.benchmark"


def load_script(name):
    """Import a scripts/<name>.py module (to reuse its table functions) without running its main()."""
    import importlib.util

    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"bp_script_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_analyzer(cfg, db=None):
    from src.benchdb.analyze import Analyzer

    bc = cfg["benchdb"]
    model = (db.manifest.get("spacy_model") if db is not None else None) or bc.get("spacy_model") \
        or cfg["real_queries"]["spacy_model"]
    return Analyzer(model, concepts=(db.manifest.get("concepts", bc["concepts"]) if db is not None else bc["concepts"]),
                    answer_chars=bc["answer_chars"])


def ensure_real_db(cfg, domain):
    from src.benchdb.readers import read_real
    from src.benchdb.store import BenchDB

    path = real_db_path(cfg, domain)
    if path.exists():
        return BenchDB.open(path)
    print(f"Building the real-query database for '{domain}' ...")
    db = BenchDB.create(path, info={"source": f"real_queries/{domain}.jsonl"})
    db.add_items(read_real(cfg, domain), make_analyzer(cfg), kind="real", passage_entities=False)
    return db


def ensure_run_db(cfg, rag, methods):
    from src.benchdb.readers import read_run
    from src.benchdb.store import BenchDB

    path = run_db_path(cfg, rag)
    if path.exists():
        return BenchDB.open(path)
    print(f"Building the benchmark database for {rag} ...")
    db = BenchDB.create(path, info={"source": f"benchmark run of {rag}"})
    db.add_items(read_run(cfg, rag, methods), make_analyzer(cfg), kind="benchmark",
                 passage_entities=cfg["benchdb"]["passage_entities"])
    return db
