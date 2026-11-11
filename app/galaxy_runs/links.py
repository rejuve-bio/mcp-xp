"""Links that open Galaxy's own forms with a card's values filled in ("Open in Galaxy")."""

import re
from typing import Any, Dict, List, Tuple
from urllib.parse import quote

_LATER_REPEAT_BLOCK = re.compile(r"_([1-9]\d*)\|")


def tool_form_url(galaxy_url: str, tool_id: str, version: str, history_id: str,
                  inputs: Dict[str, Any]) -> Tuple[str, List[str]]:
    """Galaxy's tool form passes every query value to its build, as flat 'a|b' keys.

    Returns the url and the paths a link can't hold: collections, several files in one
    input, and repeat blocks after the first.
    """
    params: List[Tuple[str, str]] = [("tool_id", tool_id)]
    if version:
        params.append(("version", version))
    params.append(("history_id", history_id))
    not_carried: List[str] = []
    for path, value in inputs.items():
        if _LATER_REPEAT_BLOCK.search(path):
            not_carried.append(path)
        elif isinstance(value, dict) and "src" in value:
            if value.get("src") == "hda" and value.get("id"):
                params.append((path, value["id"]))
            else:
                not_carried.append(path)
        elif isinstance(value, (list, dict)):
            not_carried.append(path)
        elif value is None:
            continue
        else:
            params.append((path, _scalar(value)))
    query = "&".join(f"{_enc(k)}={_enc(v)}" for k, v in params)
    return f"{galaxy_url.rstrip('/')}/?{query}", not_carried


def rerun_url(galaxy_url: str, job_id: str) -> str:
    """Galaxy's 'run again' form, filled from the job (collections included)."""
    return f"{galaxy_url.rstrip('/')}/?job_id={_enc(job_id)}"


def workflow_landing_url(galaxy_url: str, uuid: str) -> str:
    return f"{galaxy_url.rstrip('/')}/workflow_landings/{_enc(uuid)}"


def _scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _enc(value: str) -> str:
    # Encode everything: '|' as %7C and '+' as %2B, since Galaxy reads a bare '+' as a space.
    return quote(str(value), safe="")
