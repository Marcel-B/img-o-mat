"""Cross-runner compatibility: PipeNetwork 4-bit checkpoint -> Argus single-block loader.

Optional on Apple Silicon. Both source directories must point to the pinned revisions.
Each runner executes in its own process, just as the production phase boundary does.
"""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = r'''
import json, sys
from pathlib import Path
from dataclasses import asdict
source, mode, destination, worker = sys.argv[1:]
sys.path.insert(0, source)
import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten
from minimax_h3_mlx.adaln import ModulationCache
root = Path(destination)
if mode == "reference":
    from minimax_h3_mlx.config import DiTConfig, TAG_AUDIO, TAG_TEXT, TAG_VIDEO
    from minimax_h3_mlx.dit import MiniMaxH3DiT
    from minimax_h3_mlx.quantize import QuantConfig, quantize_dit
    mx.random.seed(17)
    cfg = DiTConfig(hidden_size=256, num_layers=3, token_refiner_num_layers=1,
                    num_attention_heads=4, attention_head_dim=64, ffn_hidden_size=128,
                    latents_dim=4, audio_latents_dim=8, text_dim=128, timestep_input_dim=16,
                    time_embed_hidden_size=256, time_embed_dim=64, adaln_out_features=4608,
                    final_adaln_out_features=512, rope_inv_freq_len=4)
    model = MiniMaxH3DiT(cfg)
    quantize_dit(model, QuantConfig(bits=4, group_size=64, quantize_adaln=True, adaln_bits=8))
    state = dict(tree_flatten(model.parameters()))
    mx.save_safetensors(str(root / "model.safetensors"), state)
    (root / "config.json").write_text(json.dumps(asdict(cfg)))
    (root / "quant_config.json").write_text(json.dumps({"bits":4,"group_size":64,"quantize_adaln":True,"adaln_bits":8}))
    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map":{k:"model.safetensors" for k in state}}))
    text, video, audio = 4, 8, 4
    sequence = text + video + audio
    args = [mx.random.normal((1,video,cfg.video_patch_dim)), mx.random.normal((1,audio,8)),
            mx.random.normal((1,text,128)), mx.array([0.0,0.7]),
            mx.array([0]*text+[1]*video+[0]*audio),
            mx.array([TAG_TEXT]*text+[TAG_VIDEO]*video+[TAG_AUDIO]*audio),
            mx.stack([mx.arange(sequence)%3,mx.arange(sequence)%5,mx.arange(sequence)%7],axis=-1),
            mx.arange(text,text+video),mx.arange(text+video,sequence),mx.arange(text)]
    mx.save_safetensors(str(root / "inputs.safetensors"), {str(i):a for i,a in enumerate(args)})
    cache = ModulationCache.build(model, args[3], dtype=mx.bfloat16)
    v,a = model(*args,modulation_cache=cache)
    mx.save_safetensors(str(root / "expected.safetensors"), {"video":v,"audio":a})
    from types import SimpleNamespace
    from minimax_h3_mlx.pipeline import MiniMaxH3Pipeline
    from minimax_h3_mlx.config import PipelineConfig
    class Encoder:
        def encode(self, *unused): return args[2], np.ones(text, dtype=np.int64)
    class Capture(MiniMaxH3Pipeline):
        def _decode_video(self, rows, *shape): return np.array(rows)
        def _decode_audio(self, rows, *shape): return np.array(rows)
    for directory in (root / "video_vae" / "source", root / "audio_vae"):
        directory.mkdir(parents=True)
    (root / "video_vae" / "source" / "config.json").write_text(json.dumps({"space_down":[2,2,2,2],"z_channels":4}))
    (root / "video_vae" / "config.json").write_text(json.dumps({"latents_mean":[0]*4,"latents_std":[1]*4}))
    (root / "audio_vae" / "config.json").write_text(json.dumps({"latent_channels":8,"sample_rate":32000,"latents_mean":[0]*8,"latents_std":[1]*8}))
    (root / "model_index.json").write_text("{}")
    mx.save_safetensors(str(root / "conditioning.safetensors"), {"embeds":args[2]})
    np.save(root / "tags.npy", np.ones(text,dtype=np.int64))
    video_cfg=SimpleNamespace(spatial_compression_ratio=16,latent_channels=4)
    audio_cfg=SimpleNamespace(latent_channels=8,sampling_rate=32000)
    result=Capture(model,Encoder(),SimpleNamespace(config=video_cfg),SimpleNamespace(config=audio_cfg),PipelineConfig())(
        "fox",duration_seconds=5,num_inference_steps=3,seed=3,height=32,width=32,verbose=False)
    mx.save_safetensors(str(root / "expected-latents.safetensors"), {"video":mx.array(result.video),"audio":mx.array(result.audio)})
    (root / "request.json").write_text(json.dumps({"checkpoint":str(root),"transformer":str(root),"prompt":"fox",
        "image":[],"anchor":[],"duration":5,"steps":3,"seed":3,"width":32,"height":32}))
else:
    from minimax_h3_mlx.streaming import load_streaming_dit
    model,provider = load_streaming_dit(root,stream_block_group_size=1)
    assert len(provider.slots) == 1
    assert provider.block_count == 3
    assert not any(k.startswith("blocks.") for k,_ in tree_flatten(model.parameters()))
    inputs = mx.load(str(root / "inputs.safetensors"))
    args = [inputs[str(i)] for i in range(10)]
    expected = mx.load(str(root / "expected.safetensors"))
    cache = ModulationCache.build_streaming(model,provider,args[3],dtype=mx.bfloat16)
    for _ in range(2):
        v,a = model(*args,modulation_cache=cache,block_provider=provider)
        mx.eval(v,a)
        np.testing.assert_array_equal(np.array(v),np.array(expected["video"]))
        np.testing.assert_array_equal(np.array(a),np.array(expected["audio"]))
        assert provider.current_group_end - provider.current_group_start == 1
    sys.path.insert(0,worker)
    import minimax_low_ram
    minimax_low_ram.run_stage("sample",json.loads((root / "request.json").read_text()),root)
    expected_latents=mx.load(str(root / "expected-latents.safetensors"))
    for name in ["video","audio"]:
        actual=mx.load(str(root / (name+"-latents.safetensors")))["rows"]
        np.testing.assert_array_equal(np.array(actual),np.array(expected_latents[name]))
    print(json.dumps({"residentBlockSlots":len(provider.slots),"totalBlocks":provider.block_count,
                      "videoMaxError":float(mx.max(mx.abs(v-expected["video"])).item()),
                      "audioMaxError":float(mx.max(mx.abs(a-expected["audio"])).item())}))
'''


@unittest.skipUnless(importlib.util.find_spec("mlx") and os.environ.get("IMG_O_MAT_MINIMAX_SOURCE")
                     and os.environ.get("IMG_O_MAT_MINIMAX_STREAMING_SOURCE"),
                     "requires MLX and both pinned runner source directories")
class StreamingCompatibilityTest(unittest.TestCase):
    def test_original_quantized_checkpoint_streams_with_identical_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = root / "check.py"
            script.write_text(SCRIPT)
            for mode, key in (("reference", "IMG_O_MAT_MINIMAX_SOURCE"),
                              ("stream", "IMG_O_MAT_MINIMAX_STREAMING_SOURCE")):
                result = subprocess.run([sys.executable, str(script), os.environ[key], mode, str(root),
                                         str(Path(__file__).resolve().parents[1] / "worker")],
                                        capture_output=True, text=True, timeout=120)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            values = json.loads(result.stdout.strip().splitlines()[-1])
            self.assertEqual(values["residentBlockSlots"], 1)
            self.assertEqual(values["totalBlocks"], 3)
            self.assertEqual(values["videoMaxError"], 0)
            self.assertEqual(values["audioMaxError"], 0)


if __name__ == "__main__":
    unittest.main()
