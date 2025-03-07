import litserve as ls
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

class Qwen25VLAPI(ls.LitAPI):
    def setup(self, device, model_name="Qwen/Qwen2.5-VL-7B-Instruct-AWQ"):
        """
        Initializes the Qwen2.5 VL model and processor.
        """
        self.device = device
        self.dtype = torch.float16
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            attn_implementation="flash_attention_2",
            device_map=self.device
        )

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.prompt = "Describe this image"

    def decode_request(self, request):

        image_base64 = request["image_base64"]

        return image_base64


    def predict(self, image_base64: str):
        """
        Takes a base64-encoded image and generates a caption.
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"data:image;base64,{image_base64}"},  # Pass base64 directly
                    {"type": "text", "text": self.prompt},
                ],
            }
        ]

        # Prepare inputs
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        # Generate output
        generated_ids = self.model.generate(**inputs, max_new_tokens=128) # default is 128
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return [output_text[0]]

    def encode_response(self, output):
        return {"description": output[0]}

if __name__ == "__main__":
    api = Qwen25VLAPI()
    server = ls.LitServer(api, accelerator="gpu", api_path="/caption_image")
    server.run(port=8000)
