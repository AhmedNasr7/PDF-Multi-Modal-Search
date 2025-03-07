import base64
from pathlib import Path
from docling_core.types.doc import PictureItem, TextItem
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from modules.vlm_service import VLMService  # Importing the VLM service

class DocumentParser:
    """
    Parses PDF documents, extracts text & images, and outputs structured JSON.
    """

    def __init__(self, vlm_service: VLMService):
        """
        Initializes the parser with a VLM service instance.
        """
        self.vlm_service = vlm_service  # Inject VLM service for captioning
        self.output_dir = Path("output")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Docling configuration
        self.pipeline_options = PdfPipelineOptions()
        self.pipeline_options.images_scale = 2  # Adjust scale
        self.pipeline_options.generate_page_images = True
        self.pipeline_options.generate_picture_images = True

        self.doc_converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=self.pipeline_options)}
        )

    @staticmethod
    def encode_image_to_base64(image_path):
        """Encodes an image file to a base64 string."""
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")

    def parse_pdf(self, pdf_path):
        """
        Parses a PDF and extracts structured text & images.
        :param pdf_path: Path to the PDF file.
        :return: Dictionary containing structured document data.
        """
        pdf_path = Path(pdf_path)
        conv_res = self.doc_converter.convert(pdf_path)
        doc_filename = pdf_path.stem
        structured_data = {"document": doc_filename, "content": []}
        picture_counter = 0

        for element, _level in conv_res.document.iterate_items():
            if isinstance(element, TextItem):
                structured_data["content"].append({"type": "text", "text": element.text.strip()})

            elif isinstance(element, PictureItem):
                picture_counter += 1
                image_path = self.output_dir / f"{doc_filename}-picture-{picture_counter}.png"
                element.get_image(conv_res.document).save(image_path, "PNG")
                base64_img = self.encode_image_to_base64(image_path)

                # Generate real caption using VLMService
                caption = self.vlm_service.generate_caption(base64_img, prompt="Describe this image.")

                structured_data["content"].append({
                    "type": "image",
                    "index": picture_counter,
                    "image_base64": base64_img,
                    "caption": caption
                })

        return structured_data
