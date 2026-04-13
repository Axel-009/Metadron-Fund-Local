from modeling_qwen2_5_omni_low_VRAM_mode import Qwen2_5OmniForConditionalGeneration
from transformers import Qwen2_5OmniProcessor
from transformers.utils.hub import cached_file
from gptqmodel import GPTQModel
from gptqmodel.models.base import BaseGPTQModel
from gptqmodel.models.auto import MODEL_MAP
from gptqmodel.models._const import CPU, SUPPORTED_MODELS
from huggingface_hub import snapshot_download
from qwen_omni_utils import process_mm_info
import torch, time, soundfile as sf

model_path = "Qwen/Qwen2.5-Omni-7B-GPTQ-Int4"
model_path = snapshot_download(repo_id=model_path)

class Qwen25OmniThinkerGPTQ(BaseGPTQModel):
    loader = Qwen2_5OmniForConditionalGeneration
    base_modules = ["thinker.model.embed_tokens", "thinker.model.norm", "token2wav", "thinker.audio_tower", "thinker.model.rotary_emb", "thinker.visual", "talker"]
    pre_lm_head_norm_module = "thinker.model.norm"
    require_monkeypatch = False
    layers_node = "thinker.model.layers"
    layer_type = "Qwen2_5OmniDecoderLayer"
    layer_modules = [["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"], ["self_attn.o_proj"], ["mlp.up_proj", "mlp.gate_proj"], ["mlp.down_proj"]]

MODEL_MAP["qwen2_5_omni"] = Qwen25OmniThinkerGPTQ
SUPPORTED_MODELS.extend(["qwen2_5_omni"])

device_map = {"thinker.model": "cuda", "thinker.lm_head": "cuda", "thinker.visual": "cpu", "thinker.audio_tower": "cpu", "talker": "cuda", "token2wav": "cuda"}
model = GPTQModel.load(model_path, device_map=device_map, torch_dtype=torch.float16, attn_implementation="flash_attention_2")
processor = Qwen2_5OmniProcessor.from_pretrained(model_path)

def video_inference(video_path, prompt, sys_prompt):
    messages = [{"role": "system", "content": [{"type": "text", "text": sys_prompt}]}, {"role": "user", "content": [{"type": "video", "video": video_path}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    audios, images, videos = process_mm_info(messages, use_audio_in_video=True)
    inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True).to('cuda').to(model.dtype)
    output = model.generate(**inputs, use_audio_in_video=True, return_audio=True)
    return processor.batch_decode(output[0], skip_special_tokens=True, clean_up_tokenization_spaces=False), output[2]

video_path = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-Omni/draw.mp4"
sys_prompt = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group."
torch.cuda.reset_peak_memory_stats()
start = time.time()
response, audio = video_inference(video_path, prompt=None, sys_prompt=sys_prompt)
end = time.time()
sf.write("./output_audio_gptq.wav", audio.reshape(-1).detach().cpu().numpy(), samplerate=24000)
print(response[0])
print(f"Time: {end-start:.2f}s, Peak GPU: {torch.cuda.max_memory_allocated()/1024/1024:.2f} MB")
