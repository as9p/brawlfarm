"""GET and PUT /api/settings: the whole config.toml document as JSON.

PUT validates the document, refuses to drop an instance whose worker is alive (one worker
per instance -- removing it from settings would orphan the process the supervisor is
watching), saves it atomically and hands it to the supervisor. Changing [app].port takes
effect on the next start, because uvicorn is already bound.

The Brawl Stars token and the webhook URL never leave in full: every answer carries a
masked form (eight asterisks plus the last four characters), so a page that got a request
through anyway learns nothing it can use. The panel PUTs the whole document back, masked
values included, so a PUT whose secret equals the mask of the stored one keeps the stored
secret; anything else is a new value and is saved as typed.
"""

from __future__ import annotations

import asyncio
import os
import sys

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import ValidationError

from brawlfarm import settings as S
from brawlfarm.api.deps import LIVE_STATES, get_home, get_sup

router = APIRouter()

# (section, field) of every secret the answers mask.
SECRET_FIELDS = (("connection", "brawl_api_token"), ("notifications", "webhook_url"))
MASK = "********"


def mask_secret(value: str) -> str:
    """Blank stays blank, so the panel can still tell set from unset. Anything shorter
    than twelve characters is all mask, so the tail never gives away most of it."""
    if not value:
        return ""
    return MASK + value[-4:] if len(value) >= 12 else MASK


def masked(settings: S.AppSettings) -> dict:
    """The document as JSON with every secret in SECRET_FIELDS masked."""
    doc = settings.model_dump(mode="json")
    for section, field in SECRET_FIELDS:
        doc[section][field] = mask_secret(doc[section][field])
    return doc


def keep_masked_secrets(body: dict, current: S.AppSettings) -> dict:
    """Put the stored secret back wherever the body sent its mask unchanged."""
    for section, field in SECRET_FIELDS:
        part = body.get(section)
        if not isinstance(part, dict):
            continue
        stored = getattr(getattr(current, section), field)
        if stored and part.get(field) == mask_secret(stored):
            part[field] = stored
    return body


@router.get("/api/settings")
async def read_settings(request: Request) -> dict:
    """The settings the supervisor is running with right now (not a re-read of the file)."""
    return masked(get_sup(request).settings)


@router.get("/api/settings/defaults")
async def read_settings_defaults() -> dict:
    """Every section at its model default, with no instances. Read only: nothing is saved
    and nothing is applied, so the panel can show what a switch would go back to."""
    return S.AppSettings().model_dump(mode="json")


@router.put("/api/settings")
async def write_settings(request: Request, body: dict) -> dict:
    """Replace config.toml wholesale and re-apply it. 422 names the offending field, 409
    names the instance that has to be stopped first."""
    sup = get_sup(request)
    body = keep_masked_secrets(body, sup.settings)
    try:
        new = S.AppSettings.model_validate(body)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=S._explain(exc)) from exc
    keeping = {i.name for i in new.instances}
    views = {v.name: v for v in sup.views()}
    for inst in sup.settings.instances:
        if inst.name in keeping:
            continue
        view = views.get(inst.name)
        if view is not None and view.state in LIVE_STATES:
            raise HTTPException(status_code=409, detail=f"Stop {inst.name} before removing it")
    S.save(new, get_home(request))
    sup.apply_settings(new)
    sup.poke()
    return masked(new)


@router.post("/api/settings/reset")
async def reset_settings(request: Request) -> dict:
    """Every section back to its model default, with the instances list carried over.

    A reset that also emptied the fleet would be a factory reset of somebody's farm, and it
    would orphan every data folder behind it. Because nothing is removed, the 409 that
    guards PUT cannot happen here, and no instance has to be stopped first.
    """
    sup = get_sup(request)
    new = S.AppSettings(instances=list(sup.settings.instances))
    S.save(new, get_home(request))
    sup.apply_settings(new)
    sup.poke()
    return masked(new)


@router.post("/api/settings/open-data-folder", status_code=204)
async def open_data_folder(request: Request) -> Response:
    """Open the data directory in Explorer.

    Windows only, and the guard is on sys.platform rather than on a try around os.startfile,
    because that attribute does not exist anywhere else. startfile can block on a busy shell,
    so it goes through a thread. The path is never put in the answer: the panel shows it in
    one tooltip and nowhere else, and it is a Windows user path.
    """
    if sys.platform != "win32":
        raise HTTPException(status_code=501, detail="Only on Windows")
    try:
        await asyncio.to_thread(os.startfile, str(get_home(request)))
    except OSError as exc:
        raise HTTPException(status_code=500, detail="could not open the data folder") from exc
    return Response(status_code=204)
