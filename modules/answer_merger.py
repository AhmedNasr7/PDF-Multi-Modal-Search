from transformers import T5ForConditionalGeneration, T5Tokenizer


# Optional: I might use T5 To generate summarized coherent answers from multiple retrieved answers. 

class T5AnswerMerger:
    """
    Uses T5 to generate a concise and structured answer from multiple retrieved text snippets.
    """

    def __init__(self, model_name="google/flan-t5-large" ):
        """
        Initializes the T5 transformer model.
        :param model_name: Pre-trained model
        """
        # model_name = "google/flan-t5-large" 
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)


    def generate_text(self, prompt, max_length=500):
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        output = self.model.generate(input_ids, max_length=max_length, num_return_sequences=1)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)


    def merge_and_summarize(self, chunks, max_length=2000, min_length=10):
        """
        Merges multiple retrieved text chunks and summarizes them using T5.
        :param chunks: List of text snippets.
        :param max_length: Maximum token length of the summarized output.
        :param min_length: Minimum length of the summary.
        :return: Summarized answer.
        """
        merged_text = " ".join(chunks)  # Merge all retrieved chunks into a single text block

        # Prepend summarization task (T5 requires a task-specific prefix)
        input_text = "Summarize this text: " + merged_text  

        outputs = self.generate_text(input_text)
        
        return outputs
