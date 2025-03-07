from transformers import pipeline

# Optional: I might use T5 To generate coherent answer from multiple retrieved answers. 

class T5AnswerMerger:
    """
    Uses T5 to generate a concise and structured answer from multiple retrieved text snippets.
    """

    def __init__(self, model_name="t5-large"):
        """
        Initializes the T5 transformer model.
        :param model_name: Pre-trained T5 model (e.g., "t5-base", "t5-large").
        """
        self.summarizer = pipeline("summarization", model=model_name, tokenizer=model_name)

    def merge_and_summarize(self, chunks, max_length=250, min_length=50):
        """
        Merges multiple retrieved text chunks and summarizes them using T5.
        :param chunks: List of text snippets.
        :param max_length: Maximum token length of the summarized output.
        :param min_length: Minimum length of the summary.
        :return: Summarized answer.
        """
        merged_text = " ".join(chunks)  # Merge all retrieved chunks into a single text block

        # Prepend summarization task (T5 requires a task-specific prefix)
        input_text = "summarize: " + merged_text  

        summarized_text = self.summarizer(
            input_text, max_length=max_length, min_length=min_length, do_sample=False
        )

        return summarized_text[0]['summary_text']
