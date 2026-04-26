from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

try:
    from qwen_vl_utils import process_vision_info
except Exception as exc:  # pragma: no cover
    process_vision_info = None
    _QWEN_VL_UTILS_IMPORT_ERROR = exc
else:
    _QWEN_VL_UTILS_IMPORT_ERROR = None


@dataclass
class QwenVLConfig:
    model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    device_map: str = "auto"
    torch_dtype: str = "auto"


class Qwen25VLWrapper:
    """Batch-size-1 inference wrapper for Qwen2.5-VL-Instruct."""

    def __init__(self, config: QwenVLConfig | None = None):
        self.config = config or QwenVLConfig()
        self.processor = AutoProcessor.from_pretrained(self.config.model_name)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.config.model_name,
            torch_dtype=self.config.torch_dtype,
            device_map=self.config.device_map,
        )
        self.model.eval()

    def _build_messages(self, question: str, image: Optional[Image.Image] = None) -> list[dict]:
        if image is None:
            content = [{"type": "text", "text": question}]
        else:
            content = [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ]
        return [{"role": "user", "content": content}]

    @torch.inference_mode()
    def generate(
        self,
        question: str,
        image: Optional[Image.Image] = None,
        max_new_tokens: int = 16,
        do_sample: bool = False,
        temperature: float = 0.0,
    ) -> str:
        messages = self._build_messages(question=question, image=image)
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        if image is None:
            inputs = self.processor(
                text=[text],
                padding=True,
                return_tensors="pt",
            )
        else:
            if process_vision_info is None:
                raise ImportError(
                    "qwen-vl-utils is required for image inference. "
                    f"Original error: {_QWEN_VL_UTILS_IMPORT_ERROR}"
                )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )

        inputs = inputs.to(self.model.device)

        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
        }
        # For greedy decoding, temperature is intentionally omitted to avoid
        # Transformers warnings and accidental sampling behavior.
        if do_sample:
            generation_kwargs["temperature"] = temperature

        generated_ids = self.model.generate(**inputs, **generation_kwargs)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return output_text.strip()
