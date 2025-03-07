import requests

class VLMService:
    """
    Handles communication with the Vision-Language Model (Qwen2.5-VL).
    """

    def __init__(self, api_url="http://localhost:8000/caption_image"):
        """
        Initializes the VLMService with the given API endpoint.
        :param api_url: The URL of the VLM model server.
        """
        self.api_url = api_url

    def generate_caption(self, image_base64, prompt="Describe this image."):
        """
        Sends a base64-encoded image to the VLM API for captioning.
        
        :param image_base64: Base64-encoded image string.
        :param prompt: Captioning prompt (default: "Describe this image.").
        :return: The generated caption or an error message.
        """
        payload = {"image_base64": image_base64, "prompt": prompt}

        try:
            response = requests.post(self.api_url, json=payload)

            if response.status_code == 200:
                return response.json()
            else:
                return f"Error: {response.status_code}, {response.text}"

        except requests.exceptions.RequestException as e:
            return f"API request failed: {e}"


if __name__ == "__main__":
    from modules.vlm_service import VLMService

    api_url = "http://localhost:8000/caption_image"
    vlm_service = VLMService(api_url=api_url)

    image_base64 = ""
    caption = vlm_service.generate_caption(image_base64)
    print("Generated Caption:", caption)
