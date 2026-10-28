"""The image models the service offers, and whether this Mac can paint with each.

Moved from Tonwerk-AI's Images/ImageModels.cs (2026-10-28) with the same ids, so pictures Tonwerk painted before keep
naming their model. The list can be replaced in the configuration ("models"); an entry there takes the defaults' fields
it leaves out.
"""

from __future__ import annotations

MFLUX = "mflux"
FLUX3 = "flux3-mlx"

# How many reference pictures (style, figure, place) a model that looks at pictures takes; Tonwerk's cover styles hold
# four too.
MAX_REFERENCES = 4

# Falconsai's NSFW classifier, which paint.py downloads into the same cache the first time a filtered model paints.
FILTER_REPO = "Falconsai/nsfw_image_detection"

FIELDS = {
    "id": "",
    "label": "",
    "runner": MFLUX,
    # The runner's name for it: mflux's model name, which picks configuration and repository, or flux3-mlx's build.
    "model": "",
    "repo": "",
    # Further repositories its first picture downloads (FLUX 3's prompt upsampler).
    "moreRepos": [],
    # Bits mflux quantizes to when it loads (the download is the full one either way); null for bfloat16.
    "quantize": None,
    # The steps the model is distilled to; more make it slower, not better.
    "steps": 4,
    # flux3-mlx's prompt upsampler; FLUX 3 Klein was trained on its captions and Black Forest Labs say never without.
    "upsampler": None,
    # Every picture passes Falconsai's classifier before it is answered: Krea 2's licence (§4.2) requires a content
    # filter, FLUX 3's and Klein 9B's (§2.e) a filter or a review before publishing. No request can switch it off.
    "contentFilter": False,
    # A Mac with less memory never paints it (Marcel's wish for Krea 2 and FLUX 3, 2026-10-21).
    "minSystemGb": None,
    # It can write short words into the picture.
    "lettering": False,
    "downloadGb": 0,
    # Roughly what it holds while painting; the LLM gate checks free memory by it.
    "memoryGb": 0,
    "license": "",
    # Whether its pictures may be used commercially.
    "commercial": False,
    # Gated repository: a Hugging Face token of an account that accepted the licence is needed.
    "gated": False,
}

# Sizes are what model cards and READMEs name; memory is an estimate, not measured.
DEFAULTS = [
    {
        "id": "klein-4b",
        "label": "FLUX.2 Klein 4B",
        "model": "flux2-klein-4b",
        "repo": "black-forest-labs/FLUX.2-klein-4B",
        "quantize": 8,
        "downloadGb": 15,
        "memoryGb": 9,
        "license": "Apache 2.0",
        "commercial": True,
    },
    {
        "id": "klein-9b",
        "label": "FLUX.2 Klein 9B",
        "model": "flux2-klein-9b",
        "repo": "black-forest-labs/FLUX.2-klein-9B",
        # At 8 bits it would hold about 18 GB, too much beside macOS on 24 GB.
        "quantize": 4,
        "downloadGb": 32,
        "memoryGb": 10,
        "license": "FLUX Non-Commercial",
        "contentFilter": True,
        "gated": True,
    },
    {
        "id": "z-image-turbo",
        "label": "Z-Image Turbo",
        "model": "z-image-turbo",
        "repo": "Tongyi-MAI/Z-Image-Turbo",
        "quantize": 8,
        # Distilled to eight denoising steps, which take nine scheduler steps; mflux's README uses nine.
        "steps": 9,
        "lettering": True,
        "downloadGb": 21,
        "memoryGb": 11,
        "license": "Apache 2.0",
        "commercial": True,
    },
    {
        "id": "krea-2-turbo",
        "label": "Krea 2 Turbo",
        "model": "krea-2",
        "repo": "krea/Krea-2-Turbo",
        "quantize": 8,
        "steps": 8,
        "contentFilter": True,
        "minSystemGb": 32,
        "lettering": True,
        "downloadGb": 35,
        "memoryGb": 22,
        "license": "Krea 2 Community (kommerziell bis 1 Mio. $ Umsatz)",
        "commercial": True,
        "gated": True,
    },
    {
        "id": "flux3-klein",
        "label": "FLUX 3 Klein",
        "runner": FLUX3,
        "model": "8bit",
        "repo": "VisualInference/FLUX-3-Image-Klein-MLX-8bit",
        "moreRepos": ["VisualInference/FLUX-3-Image-Klein-pup-MLX-4bit"],
        "steps": 16,
        "upsampler": "4bit",
        "contentFilter": True,
        "minSystemGb": 32,
        "lettering": True,
        "downloadGb": 15,
        "memoryGb": 12,
        "license": "FLUX Kommunity (kommerziell bis 5 Mio. $ Umsatz)",
        "commercial": True,
    },
]


def complete(entry: dict) -> dict:
    model = {**FIELDS, **entry}
    model["moreRepos"] = list(model.get("moreRepos") or [])
    return model


def takes_loras(model: dict) -> bool:
    """mflux applies LoRAs for Klein, Z-Image and Krea 2; flux3-mlx has no loader."""
    return model["runner"] == MFLUX


def max_references(model: dict) -> int:
    """FLUX.2 Klein looks at pictures through mflux's edit pipeline, FLUX 3 through flux3-mlx's images."""
    return MAX_REFERENCES if model["runner"] == FLUX3 or model["model"].startswith("flux2-klein") else 0


def takes_start_image(model: dict) -> bool:
    """Z-Image and Krea 2 paint over a start picture instead (img2img), keeping its layout by a strength."""
    return model["runner"] == MFLUX and model["model"].startswith(("z-image", "krea-2"))


def repos(model: dict) -> list[str]:
    extra = [FILTER_REPO] if model["contentFilter"] else []
    return [model["repo"], *model["moreRepos"], *extra]


class Catalogue:
    def __init__(self, config):
        self.config = config
        entries = config.get("models") or DEFAULTS
        self.models = [complete(entry) for entry in entries if entry.get("id")]

    def find(self, model_id: str) -> dict | None:
        return next((m for m in self.models if m["id"] == model_id), None)

    def all_repos(self) -> list[str]:
        return sorted({repo for model in self.models for repo in repos(model)})

    def unavailable(self, model: dict, probe: dict | None, probe_error: str | None) -> str | None:
        """Why the Mac that paints cannot paint with the model; None when it can."""
        if probe is None:
            return probe_error or "Der Mac, der malt, antwortet nicht."
        configured = self.config.get("systemMemoryGb")
        has = float(configured) if configured else probe.get("memoryGb")
        # Half a GB of slack: what the system reports can fall a little short of what the Mac is sold with.
        if model["minSystemGb"] and has is not None and has < model["minSystemGb"] - 0.5:
            return f"{model['label']} braucht einen Mac mit {model['minSystemGb']:.0f} GB, {probe.get('host') or 'dieser'} hat {has:.0f}."
        if not (probe.get("runners") or {}).get(model["runner"]):
            return f"{model['runner']} ist auf {probe.get('host') or 'dem Mac'} nicht installiert; dort install.sh --worker ausführen."
        return None

    def downloaded(self, model: dict, probe: dict | None) -> bool | None:
        if probe is None:
            return None
        have = probe.get("downloaded") or {}
        return all(have.get(repo) for repo in repos(model))

    def describe(self, model: dict, probe: dict | None, probe_error: str | None, download: dict | None = None) -> dict:
        """An entry of GET /v1/models: OpenAI's fields plus what a client needs to offer the model."""
        reason = self.unavailable(model, probe, probe_error)
        return {
            "id": model["id"],
            "object": "model",
            "owned_by": "img-o-mat",
            "label": model["label"],
            "runner": model["runner"],
            "license": model["license"],
            "commercial": model["commercial"],
            "gated": model["gated"],
            "tokenFound": probe.get("tokenFound") if probe else None,
            "contentFilter": model["contentFilter"],
            "minSystemGb": model["minSystemGb"],
            "downloadGb": model["downloadGb"],
            "memoryGb": model["memoryGb"],
            "steps": model["steps"],
            "lettering": model["lettering"],
            "loras": takes_loras(model),
            "references": max_references(model),
            "startImage": takes_start_image(model),
            "downloaded": self.downloaded(model, probe),
            "download": download,
            "available": reason is None,
            "unavailable": reason,
        }
