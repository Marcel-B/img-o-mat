"""Paints one picture for img-o-mat, with mflux or flux3-mlx. Moved from Tonwerk-AI's Images/yueui_image.py.

Run with the runner's own Python environment (install.sh --worker): mflux pins mlx, transformers and torch versions the
others do not share, and flux3-mlx brings mlx-vlm for its upsampler. The service starts it once per picture, over SSH on
the Mac that paints or as a child process on its own; the model is loaded for this one picture and freed with the
process, since MLX's Metal cache only goes with it.

In: one JSON object on stdin
    {"runner": "mflux" | "flux3-mlx", "model": "flux2-klein-4b", "label": "FLUX.2 Klein 4B", "creator": "Tonwerk-AI",
     "quantize": 8 | null, "steps": 4, "upsampler": "4bit" | null, "contentFilter": false, "prompt": "...",
     "seed": 42, "width": 1024, "height": 1024,
     "loras": [{"file": "<sha256>", "strength": 1.0}],
     "references": [{"file": "<sha256>", "role": "style" | "figure" | "place" | "start", "strength": 0.5 | null}]}
    Files are named by their SHA-256 in the worker's files folder, where the service copied them. References are
    pictures FLUX.2 Klein (mflux's edit pipeline) and FLUX 3 look at; the client has written what each is for into the
    prompt as "image 1", "image 2", ... A "start" picture is one Z-Image or Krea 2 paints over (img2img).
Out: lines starting with "IMGOMAT " on stdout, each a JSON event:
    {"event": "stage", "stage": "loading" | "painting" | "checking", "fraction": 0.5}
    {"event": "done", "jpeg": "<base64>", "loadSeconds": 12.3, "paintSeconds": 30.1, "peakMemoryGb": 8.9,
     "paintedPrompt": "..." | null}
Anything else is the runner's own output, which the service keeps for the error message.
"""

import base64
import io
import json
import os
import sys
import time
from xml.sax.saxutils import escape

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
from common import emit  # noqa: E402

# Krea's licence asks for a content filter, FLUX 3's and Klein 9B's for a filter or a review; this is one of the classifiers Krea's
# names, the one the ComfyUI node and Black Forest Labs' FLUX 3 reference code use too.
NSFW_REPO = "Falconsai/nsfw_image_detection"
NSFW_THRESHOLD = 0.85

GATED = (
    "Error: Hugging Face refused the download. Accept the model's licence on huggingface.co and put an"
    " access token of that account into ~/.config/tonwerk/huggingface.token on the Mac that paints."
)


def peak_memory_gb() -> float | None:
    try:
        import mlx.core as mx

        return round(mx.get_peak_memory() / 1e9, 2)
    except Exception:  # an older mlx only has it under mx.metal
        return None


def refused(error: Exception) -> bool:
    """A gated repository answers 401/403 until its licence is accepted with the token's account."""
    return type(error).__name__ in ("GatedRepoError", "RepositoryNotFoundError") or "401" in str(error) or "403" in str(error)


def missing(name: str) -> int:
    print(f"Error: the Python package '{name}' is missing; run install.sh --worker again.", flush=True)
    return 1


class Progress:
    """mflux calls every registered object that has call_in_loop once per denoising step."""

    def __init__(self, steps: int):
        self.steps = max(steps, 1)

    def call_in_loop(self, t, seed, prompt, latents, config, time_steps, **_) -> None:
        emit({"event": "stage", "stage": "painting", "fraction": round(min((t + 1) / self.steps, 1), 3)})


def resolve(request: dict) -> None:
    """The files' paths in the worker's folder, from their hashes."""
    folder = common.settings()["files"]
    for item in (request.get("loras") or []) + (request.get("references") or []):
        if item.get("file"):
            item["path"] = os.path.join(folder, item["file"])
            # Used: probe.py keeps it another month.
            os.utime(item["path"])


def looked_at(request: dict) -> list[str]:
    return [r["path"] for r in request.get("references") or [] if r.get("role") != "start"]


def start_picture(request: dict) -> dict | None:
    return next((r for r in request.get("references") or [] if r.get("role") == "start"), None)


def load_mflux(request: dict):
    """The model's class by family; all take the same arguments and paint through generate_image."""
    from mflux.models.common.config import ModelConfig

    name = request["model"]
    if name.startswith("z-image"):
        from mflux.models.z_image import ZImage as Model
    elif name.startswith("krea-2"):
        from mflux.models.krea2 import Krea2 as Model
    elif looked_at(request):
        # The same weights; the edit pipeline adds the references' latents as tokens beside the picture's.
        from mflux.models.flux2.variants import Flux2KleinEdit as Model
    else:
        from mflux.models.flux2.variants import Flux2Klein as Model
    # LoRAs are applied when the weights load (mflux bakes them into the quantized layers), one strength each.
    loras = request.get("loras") or []
    return Model(
        model_config=ModelConfig.from_name(name),
        quantize=request.get("quantize"),
        lora_paths=[lora["path"] for lora in loras] or None,
        lora_scales=[float(lora["strength"]) for lora in loras] or None,
    )


def paint_mflux(model, request: dict, steps: int):
    model.callbacks.register(Progress(steps))
    arguments = dict(
        seed=int(request["seed"]),
        prompt=request["prompt"],
        num_inference_steps=steps,
        width=int(request.get("width") or 1024),
        height=int(request.get("height") or 1024),
    )
    # Distilled Klein and Krea 2 Turbo sample at guidance 1; Z-Image Turbo takes none and picks its own.
    if not request["model"].startswith("z-image"):
        arguments["guidance"] = 1.0
    if looked_at(request):
        arguments["image_paths"] = looked_at(request)
    elif start := start_picture(request):
        # mflux skips that share of the steps, so more strength keeps more of the start picture.
        arguments["image_path"] = start["path"]
        arguments["image_strength"] = float(start.get("strength") or 0.5)
    return model.generate_image(**arguments).image, None


def load_flux3(request: dict):
    from flux3_mlx import Flux3Klein

    # low_memory: the upsampler (a 9B model at 4 bits) leaves before the transformer loads, so the picture holds about
    # 12 GB rather than both at once.
    return Flux3Klein.from_pretrained(request["model"], upsampler=request.get("upsampler"), low_memory=True)


def paint_flux3(pipe, request: dict, steps: int):
    def step(done: int, total: int) -> None:
        emit({"event": "stage", "stage": "painting", "fraction": round(min(done / max(total, 1), 1), 3)})

    from PIL import Image

    references = [Image.open(path).convert("RGB") for path in looked_at(request)]
    image, record = pipe(
        request["prompt"],
        images=references or None,
        width=int(request.get("width") or 1024),
        height=int(request.get("height") or 1024),
        steps=steps,
        seed=int(request["seed"]),
        upsample=bool(request.get("upsampler")),
        callback=step,
    )
    # What the upsampler wrote is what the model painted; the dialog shows it under the picture.
    return image, record.get("prompt") if record.get("upsampled") else None


XMP_NAMESPACE = b"http://ns.adobe.com/xap/1.0/\x00"


def with_ai_note(jpeg: bytes, model: str, creator: str = "img-o-mat") -> bytes:
    """The picture with an XMP packet saying it is AI-made, as the EU's AI Act (Art. 50(2)) asks of a generating
    system's pictures: IPTC's DigitalSourceType trainedAlgorithmicMedia, which photo apps and platforms read, plus the
    model as creator tool. Put in as its own APP1 segment after the JFIF header, so no Pillow version matters."""
    name = escape(model, {'"': "&quot;"})
    tool = escape(creator, {'"': "&quot;"})
    xmp = (
        '<?xpacket begin="\ufeff" id="W5M0MpCehiHzreSzNTczkc9d"?>'
        '<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        '<rdf:Description rdf:about=""'
        ' xmlns:Iptc4xmpExt="http://iptc.org/std/Iptc4xmpExt/2008-02-29/"'
        ' xmlns:xmp="http://ns.adobe.com/xap/1.0/"'
        ' xmlns:dc="http://purl.org/dc/elements/1.1/"'
        ' Iptc4xmpExt:DigitalSourceType="http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"'
        f' xmp:CreatorTool="{tool} ({name})">'
        f'<dc:description><rdf:Alt><rdf:li xml:lang="x-default">KI-generiert mit {name} ({tool})</rdf:li>'
        "</rdf:Alt></dc:description></rdf:Description></rdf:RDF></x:xmpmeta>"
        '<?xpacket end="w"?>'
    ).encode("utf-8")
    payload = XMP_NAMESPACE + xmp
    segment = b"\xff\xe1" + (len(payload) + 2).to_bytes(2, "big") + payload
    at = 2
    if jpeg[2:4] == b"\xff\xe0":
        at = 4 + int.from_bytes(jpeg[4:6], "big")
    return jpeg[:at] + segment + jpeg[at:]


def flagged(image) -> bool:
    """Falconsai's ViT on the CPU through transformers (mflux brings it, flux3-mlx's mlx-vlm too, torch is installed
    beside); a few hundred MB, cached in HF_HOME."""
    from transformers import pipeline

    classify = pipeline("image-classification", model=NSFW_REPO, device="cpu")
    scores = {r["label"].lower(): r["score"] for r in classify(image.convert("RGB"))}
    return scores.get("nsfw", 0.0) >= NSFW_THRESHOLD


def main() -> int:
    request = json.load(sys.stdin)
    common.prepare_environment(common.settings())
    resolve(request)
    flux3 = request.get("runner") == "flux3-mlx"
    if flux3 and request.get("loras"):
        print("Error: FLUX 3 Klein takes no LoRAs; flux3-mlx has no loader for them.", flush=True)
        return 1
    if flux3 and start_picture(request):
        print("Error: FLUX 3 Klein paints over no start picture; give it the picture as a reference.", flush=True)
        return 1
    steps = int(request.get("steps") or 4)
    emit({"event": "stage", "stage": "loading", "fraction": 0})
    started = time.perf_counter()
    try:
        # Imported here so that a missing package shows up as the picture's error rather than before the first event.
        model = load_flux3(request) if flux3 else load_mflux(request)
    except ModuleNotFoundError as gone:
        return missing(gone.name)
    except Exception as error:
        if refused(error):
            print(GATED, flush=True)
            return 1
        raise
    loaded = time.perf_counter()

    emit({"event": "stage", "stage": "painting", "fraction": 0})
    try:
        image, painted = paint_flux3(model, request, steps) if flux3 else paint_mflux(model, request, steps)
    except Exception as error:
        # flux3-mlx downloads the upsampler only when it first expands a prompt.
        if refused(error):
            print(GATED, flush=True)
            return 1
        raise
    finished = time.perf_counter()
    del model

    if request.get("contentFilter"):
        emit({"event": "stage", "stage": "checking", "fraction": 1})
        try:
            if flagged(image):
                print("Error: the content filter held the picture back; describe it differently.", flush=True)
                return 1
        except ModuleNotFoundError as gone:
            return missing(gone.name)

    # PIL directly rather than the runner's save, which in mflux swallows errors and writes PNGs of several megabytes.
    # The picture goes back on stdout, so painting over SSH needs no second connection to fetch it.
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, "JPEG", quality=92)
    jpeg = with_ai_note(buffer.getvalue(), request.get("label") or request["model"], request.get("creator") or "img-o-mat")
    emit(
        {
            "event": "done",
            "jpeg": base64.b64encode(jpeg).decode("ascii"),
            "loadSeconds": round(loaded - started, 2),
            "paintSeconds": round(finished - loaded, 2),
            "peakMemoryGb": peak_memory_gb(),
            "paintedPrompt": painted,
        }
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
