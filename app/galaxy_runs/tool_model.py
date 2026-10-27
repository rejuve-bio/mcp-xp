"""Turn Galaxy's tool model (GET/POST /api/tools/{id}/build) into flat input descriptions.

Paths are Galaxy's run format: names in the input tree joined with '|', repeat blocks as
'name_0|...'. Nothing here knows about particular tools.
"""

import re
from typing import Any, Dict, List, Optional

FILE_TYPES = {"data", "data_collection"}
SKIP_TYPES = {"hidden", "hidden_data", "baseurl", "upload_dataset"}
_REPEAT_INDEX = re.compile(r"_\d+\|")


def describe_inputs(model: Dict[str, Any], full: bool = False) -> Dict[str, Any]:
    """Flatten a tool model into {"inputs": [...], "all_count": n}.

    Trimmed (the default) keeps required inputs, every conditional's choice, and file inputs.
    full=True keeps every setting, with the section it sits in, for the card's details panel.
    Inputs that apply in some branches of a conditional only carry "when".
    """
    everything = _walk(model.get("inputs", []), prefix="", section=None, when=None)
    all_count = sum(1 for e in everything if e["type"] != "repeat")
    if full:
        return {"inputs": everything, "all_count": all_count}
    return {"inputs": [e for e in everything if _keep_trimmed(e)], "all_count": all_count}


def input_types(model: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """path -> {"type", "multiple"} for every input, all branches included. Used to shape run requests.
    Repeat blocks are listed once, as block 0; look paths up with type_of()."""
    return {e["path"]: {"type": e["type"], "multiple": e.get("multiple", False)}
            for e in _walk(model.get("inputs", []), prefix="", section=None, when=None, templates=True)}


def type_of(types: Dict[str, Dict[str, Any]], path: str) -> Optional[Dict[str, Any]]:
    """Look up a path from input_types(), whichever repeat block it is in."""
    return types.get(path) or types.get(_REPEAT_INDEX.sub("_0|", path))


def _keep_trimmed(entry: Dict[str, Any]) -> bool:
    if entry["type"] in FILE_TYPES or entry["type"] == "repeat" or entry.get("selector"):
        return True
    return not entry.get("optional") and _empty(entry.get("default"))


def _empty(value: Any) -> bool:
    return value is None or value == "" or value == [] or value == {}


def _walk(inputs: List[Dict[str, Any]], prefix: str, section: Optional[str], when: Optional[Dict[str, Any]],
          templates: bool = False) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for inp in inputs:
        kind = inp.get("type") or inp.get("model_class", "")
        name = inp.get("name", "")
        if kind == "conditional":
            out.extend(_conditional(inp, prefix, section, when, templates))
        elif kind == "section":
            out.extend(_walk(inp.get("inputs", []), f"{prefix}{name}|", inp.get("title") or inp.get("label") or name, when, templates))
        elif kind == "repeat":
            out.extend(_repeat(inp, prefix, section, when, templates))
        elif kind in SKIP_TYPES or inp.get("hidden"):
            continue
        else:
            out.append(_leaf(inp, prefix, section, when))
    return out


def _leaf(inp: Dict[str, Any], prefix: str, section: Optional[str], when: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    kind = inp.get("type", "")
    entry: Dict[str, Any] = {
        "path": f"{prefix}{inp.get('name', '')}",
        "label": inp.get("label") or inp.get("name", ""),
        "type": kind,
        "optional": bool(inp.get("optional", False)),
        "default": inp.get("value"),
    }
    if section:
        entry["section"] = section
    if inp.get("multiple"):
        entry["multiple"] = True
    if kind in ("select", "genomebuild", "data_column", "drill_down") and isinstance(inp.get("options"), list):
        entry["options"] = [o[1] for o in inp["options"] if isinstance(o, (list, tuple)) and len(o) > 1]
    if kind in FILE_TYPES:
        entry["formats"] = inp.get("extensions") or []
        if kind == "data_collection" and inp.get("collection_types"):
            entry["collection_types"] = inp["collection_types"]
        matches = _matches(inp.get("options"))
        if matches:
            entry["matches"] = matches
        entry.pop("default", None)
    if when:
        entry["when"] = dict(when)
    return entry


def _matches(options: Any) -> List[Dict[str, Any]]:
    """The thread history's datasets and collections Galaxy offers for a file input."""
    if not isinstance(options, dict):
        return []
    found = []
    for src in ("hda", "hdca"):
        for o in options.get(src) or []:
            found.append({"src": src, "id": o.get("id"), "name": o.get("name"), "hid": o.get("hid")})
    return found


def _conditional(inp: Dict[str, Any], prefix: str, section: Optional[str], when: Optional[Dict[str, Any]],
                 templates: bool) -> List[Dict[str, Any]]:
    name = inp.get("name", "")
    test = inp.get("test_param") or {}
    selector = _leaf(test, f"{prefix}{name}|", section, when)
    selector["selector"] = True
    selector_path = selector["path"]
    out = [selector]

    # Merge the branches: an input found in every case applies always, otherwise it gets "when".
    cases = inp.get("cases") or []
    by_path: Dict[str, Dict[str, Any]] = {}
    seen_in: Dict[str, List[Any]] = {}
    order: List[str] = []
    for case in cases:
        value = case.get("value")
        for e in _walk(case.get("inputs", []), f"{prefix}{name}|", section, when, templates):
            if e["path"] not in by_path:
                by_path[e["path"]] = e
                seen_in[e["path"]] = []
                order.append(e["path"])
            seen_in[e["path"]].append(value)
    for path in order:
        e = by_path[path]
        values = seen_in[path]
        if len(values) < len(cases):
            cond = dict(e.get("when") or {})
            cond[selector_path] = values[0] if len(values) == 1 else values
            e["when"] = cond
        out.append(e)
    return out


def _repeat(inp: Dict[str, Any], prefix: str, section: Optional[str], when: Optional[Dict[str, Any]],
            templates: bool) -> List[Dict[str, Any]]:
    name = inp.get("name", "")
    entry: Dict[str, Any] = {
        "path": f"{prefix}{name}",
        "label": inp.get("title") or name,
        "type": "repeat",
        "min": inp.get("min", 0),
        "max": inp.get("max"),
    }
    if section:
        entry["section"] = section
    if when:
        entry["when"] = dict(when)
    out = [entry]
    title = inp.get("title") or name
    blocks = [inp.get("inputs", [])] if templates else (inp.get("cache") or [])
    for i, block in enumerate(blocks):
        out.extend(_walk(block, f"{prefix}{name}_{i}|", f"{title} {i + 1}", when, templates))
    return out


def shape_inputs(inputs: Dict[str, Any], types: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Card inputs ({src, id} files) to the shape Galaxy's run and build requests take.

    A collection given to a single-file input runs the tool once per element (Galaxy's "batch").
    """
    shaped: Dict[str, Any] = {}
    for path, value in inputs.items():
        if _is_file(value):
            t = type_of(types, path) or {}
            if t.get("type") == "data" and value.get("src") == "hdca":
                shaped[path] = {"batch": True, "values": [value]}
            else:
                shaped[path] = {"values": [value]}
        elif isinstance(value, list) and value and all(_is_file(v) for v in value):
            shaped[path] = {"values": value}
        else:
            shaped[path] = value
    return shaped


def _is_file(value: Any) -> bool:
    return isinstance(value, dict) and value.get("src") in ("hda", "hdca", "ldda") and bool(value.get("id"))
