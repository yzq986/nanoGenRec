"""Qwen3TextEmbedder + Qwen3VLEmbedder — 唯一版本。

Qwen3TextEmbedder 通过 device 参数区分单进程多卡和分布式模式:
- device=None (默认): device_map="auto"，单进程多卡模式
- device=<str>: 显式放置到指定 GPU，分布式模式
"""

import logging
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers.modeling_outputs import ModelOutput

logger = logging.getLogger(__name__)

# Constants for Qwen3-VL-Embedding configuration
QWEN_MAX_LENGTH = 8192
IMAGE_BASE_FACTOR = 16
IMAGE_FACTOR = IMAGE_BASE_FACTOR * 2
MIN_PIXELS = 4 * IMAGE_FACTOR * IMAGE_FACTOR
MAX_PIXELS = 1800 * IMAGE_FACTOR * IMAGE_FACTOR
FPS = 1
MAX_FRAMES = 64
FRAME_MAX_PIXELS = 768 * IMAGE_FACTOR * IMAGE_FACTOR
MAX_TOTAL_PIXELS = 10 * FRAME_MAX_PIXELS
PAD_TOKEN = "<|endoftext|>"


# ============================================================
# Qwen3-VL-Embedding (官方代码内嵌)
# 来源: https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B/blob/main/scripts/qwen3_vl_embedding.py
# ============================================================


@dataclass
class Qwen3VLForEmbeddingOutput(ModelOutput):
    """Output structure for embeddings"""
    last_hidden_state: Optional[torch.FloatTensor] = None
    attention_mask: Optional[torch.Tensor] = None


class Qwen3VLForEmbedding(torch.nn.Module):
    """Model class for computing embeddings from Qwen3-VL"""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Qwen3VLForEmbeddingOutput:
        # 跳过 LM head, 直接调内层 model → 只返回 last_hidden_state
        # 比 output_hidden_states=True 省掉所有中间层 (30层 × batch × seq × hidden)
        inner = getattr(self.model, 'model', self.model)
        outputs = inner(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs,
        )
        return Qwen3VLForEmbeddingOutput(
            last_hidden_state=outputs.last_hidden_state,
            attention_mask=attention_mask,
        )

    @property
    def device(self):
        return next(self.model.parameters()).device


def sample_frames(frames: List[Union[str, Image.Image]], num_segments: int, max_segments: int) -> List[str]:
    """Sample frames from video"""
    duration = len(frames)
    frame_id_array = np.linspace(0, duration - 1, num_segments, dtype=int)
    frame_id_list = frame_id_array.tolist()
    last_frame_id = frame_id_list[-1]

    sampled_frames = []
    for frame_idx in frame_id_list:
        try:
            sampled_frames.append(frames[frame_idx])
        except:
            break
    while len(sampled_frames) < num_segments:
        sampled_frames.append(frames[last_frame_id])
    return sampled_frames[:max_segments]


class Qwen3VLEmbedder:
    """官方 Qwen3-VL-Embedding 封装类"""

    def __init__(
        self,
        model_name_or_path: str,
        device: Optional[str] = None,
        max_length: int = QWEN_MAX_LENGTH,
        min_pixels: int = MIN_PIXELS,
        max_pixels: int = MAX_PIXELS,
        total_pixels: int = MAX_TOTAL_PIXELS,
        fps: float = FPS,
        num_frames: int = MAX_FRAMES,
        max_frames: int = MAX_FRAMES,
        default_instruction: str = "Represent the user's input.",
        **kwargs
    ):
        from transformers.models.qwen3_vl import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

        self.max_length = max_length
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.total_pixels = total_pixels
        self.fps = fps
        self.num_frames = num_frames
        self.max_frames = max_frames
        self.default_instruction = default_instruction

        # Qwen3-VL-Embedding config 是 bf16，显式传一下；不传 HF 会 fallback 到 fp32
        # 导致权重 + activations 都双倍显存 (2B 模型轻松吃满 40GB)
        torch_dtype = kwargs.pop('torch_dtype', torch.bfloat16)

        print(f"Loading {model_name_or_path} ({torch_dtype})...")
        if device is not None:
            # 分布式模式：显式放置到指定 GPU (torchrun per-rank)
            base_model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
                **kwargs
            ).to(device)
            self._device = torch.device(device)
            print(f"Model loaded on {self._device}")
        else:
            # 单进程多卡模式：device_map="auto"
            base_model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
                device_map="auto",
                **kwargs
            )
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Model loaded with device_map=auto")
        self.model = Qwen3VLForEmbedding(base_model)
        self.processor = Qwen3VLProcessor.from_pretrained(
            model_name_or_path, padding_side='right'
        )
        self.model.eval()

    @torch.no_grad()
    def forward(self, inputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        outputs = self.model(**inputs)
        return {
            'last_hidden_state': outputs.last_hidden_state,
            'attention_mask': inputs.get('attention_mask')
        }

    def format_model_input(
        self,
        text: Optional[str] = None,
        image: Optional[Union[str, Image.Image]] = None,
        video: Optional[Union[str, List[Union[str, Image.Image]]]] = None,
        instruction: Optional[str] = None,
        fps: Optional[float] = None,
        max_frames: Optional[int] = None
    ) -> List[Dict]:
        """Format input for the model"""
        if instruction:
            instruction = instruction.strip()
            if instruction and not unicodedata.category(instruction[-1]).startswith('P'):
                instruction = instruction + '.'

        content = []
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": instruction or self.default_instruction}]},
            {"role": "user", "content": content}
        ]

        if not text and not image and not video:
            content.append({'type': 'text', 'text': "NULL"})
            return conversation

        if video:
            video_content = None
            video_kwargs = {'total_pixels': self.total_pixels}
            if isinstance(video, list):
                video_content = video
                if self.num_frames is not None or self.max_frames is not None:
                    video_content = sample_frames(video_content, self.num_frames, self.max_frames)
                video_content = [
                    ('file://' + ele if isinstance(ele, str) else ele)
                    for ele in video_content
                ]
            elif isinstance(video, str):
                video_content = video if video.startswith(('http://', 'https://')) else 'file://' + video
                video_kwargs = {'fps': fps or self.fps, 'max_frames': max_frames or self.max_frames}
            else:
                raise TypeError(f"Unrecognized video type: {type(video)}")

            if video_content:
                content.append({'type': 'video', 'video': video_content, **video_kwargs})

        if image:
            # 支持单张图片或图片列表
            image_list = image if isinstance(image, list) else [image]
            for img in image_list:
                image_content = None
                if isinstance(img, Image.Image):
                    image_content = img
                elif isinstance(img, str):
                    image_content = img if img.startswith(('http', 'oss')) else 'file://' + img
                else:
                    continue  # 跳过无法识别的类型

                if image_content:
                    content.append({
                        'type': 'image', 'image': image_content,
                        "min_pixels": self.min_pixels,
                        "max_pixels": self.max_pixels
                    })

        if text:
            content.append({'type': 'text', 'text': text})

        return conversation

    def _preprocess_inputs(self, conversations: List[List[Dict]]) -> Dict[str, torch.Tensor]:
        """Preprocess inputs for the model"""
        import socket
        import time as _time
        from qwen_vl_utils import process_vision_info

        # 设置 socket 超时，防止图片下载卡住
        old_timeout = socket.getdefaulttimeout()
        socket.setdefaulttimeout(2)

        t0 = _time.time()
        text = self.processor.apply_chat_template(
            conversations, add_generation_prompt=True, tokenize=False
        )
        t1 = _time.time()

        try:
            images, video_inputs, video_kwargs = process_vision_info(
                conversations, image_patch_size=16,
                return_video_metadata=True, return_video_kwargs=True
            )
        except Exception as e:
            logger.error(f"Error in processing vision info: {e}")
            images = None
            video_inputs = None
            video_kwargs = {'do_sample_frames': False}
            text = self.processor.apply_chat_template(
                [{'role': 'user', 'content': [{'type': 'text', 'text': 'NULL'}]}],
                add_generation_prompt=True, tokenize=False
            )
        t2 = _time.time()

        if video_inputs is not None:
            videos, video_metadata = zip(*video_inputs)
            videos = list(videos)
            video_metadata = list(video_metadata)
        else:
            videos, video_metadata = None, None

        inputs = self.processor(
            text=text, images=images, videos=videos, video_metadata=video_metadata,
            truncation=True, max_length=self.max_length, padding=True,
            do_resize=False, return_tensors='pt', **video_kwargs
        )

        # 恢复原来的超时设置
        socket.setdefaulttimeout(old_timeout)
        return inputs

    @staticmethod
    def _pooling_last(hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Pool the last token's hidden state"""
        flipped_tensor = attention_mask.flip(dims=[1])
        last_one_positions = flipped_tensor.argmax(dim=1)
        col = attention_mask.shape[1] - last_one_positions - 1
        row = torch.arange(hidden_state.shape[0], device=hidden_state.device)
        return hidden_state[row, col]

    def process(self, inputs: List[Dict[str, Any]], normalize: bool = True) -> torch.Tensor:
        """Process inputs and generate normalized embeddings

        Args:
            inputs: List of dicts with keys 'text', 'image', 'video', 'instruction'
                   Example: [{"text": "hello"}, {"text": "world", "image": "url"}]
            normalize: Whether to L2-normalize embeddings

        Returns:
            embeddings: torch.Tensor of shape (N, 4096)
        """
        conversations = [self.format_model_input(
            text=ele.get('text'),
            image=ele.get('image'),
            video=ele.get('video'),
            instruction=ele.get('instruction'),
            fps=ele.get('fps'),
            max_frames=ele.get('max_frames')
        ) for ele in inputs]

        processed_inputs = self._preprocess_inputs(conversations)
        processed_inputs = {k: v.to(self.model.device) for k, v in processed_inputs.items()}

        outputs = self.forward(processed_inputs)
        embeddings = self._pooling_last(outputs['last_hidden_state'], outputs['attention_mask'])

        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)

        return embeddings


class Qwen3TextEmbedder:
    """Qwen3-Embedding 纯文本模型封装

    统一版本 — 通过 device 参数区分:
    - device=None (默认): device_map="auto"，单进程多卡模式
    - device=<str>: 显式放置到指定 GPU，分布式模式 (torchrun)
    """

    def __init__(
        self,
        model_name_or_path: str,
        device: Optional[str] = None,
        max_length: int = 8192,
        **kwargs
    ):
        from transformers import AutoModel, AutoTokenizer

        self.max_length = max_length

        print(f"Loading {model_name_or_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

        # 从 kwargs 提取 torch_dtype，默认 float16
        torch_dtype = kwargs.pop('torch_dtype', torch.float16)

        if device is not None:
            # 分布式模式：显式放置到指定 GPU
            self.model = AutoModel.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
                **kwargs
            ).to(device)
            self.device = torch.device(device)
            self.n_gpus = 1
        else:
            # 单进程多卡模式：device_map="auto"
            self.n_gpus = torch.cuda.device_count()
            self.model = AutoModel.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
                device_map="auto",
                **kwargs
            )
            self.device = self.model.device

        self.model.eval()
        print(f"Model loaded on {self.device}, n_gpus={self.n_gpus}")

    @torch.no_grad()
    def encode(self, texts: List[str], normalize: bool = True) -> torch.Tensor:
        """编码文本列表"""
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state[:, -1, :]

        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)

        return embeddings
