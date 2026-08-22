"""Small, versioned schema registry for the prototype."""
from __future__ import annotations

import json
from pathlib import Path
from flask import Flask, abort, jsonify

ROOT = Path(__file__).resolve().parents[1] / "configs"
app = Flask(__name__)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/schemas/<int:version>")
def schema(version: int):
    path = ROOT / f"schema_v{version}.json"
    if not path.exists(): abort(404, f"unknown schema version {version}")
    return jsonify(json.loads(path.read_text(encoding="utf-8")))


@app.get("/schemas")
def list_schemas():
    return jsonify({"versions": [int(p.stem.rsplit("v", 1)[1]) for p in sorted(ROOT.glob("schema_v*.json"))]})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8088)
