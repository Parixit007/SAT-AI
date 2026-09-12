"""In-process state for uploaded inputs and past query results. Single-process, non-persistent --
fine for a demo; swap for a real store (DB/Redis) if this needs to survive restarts or run behind
multiple workers."""

from pathlib import Path
from typing import Optional

from app.orchestrator.controller import QueryResult

_inputs: dict[str, list[Path]] = {}
_queries: dict[str, QueryResult] = {}


def save_input(input_id: str, paths: list[Path]) -> None:
    _inputs[input_id] = paths


def get_input(input_id: str) -> Optional[list[Path]]:
    return _inputs.get(input_id)


def save_query(query_id: str, result: QueryResult) -> None:
    _queries[query_id] = result


def get_query(query_id: str) -> Optional[QueryResult]:
    return _queries.get(query_id)
