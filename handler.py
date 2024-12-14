import json
import torch
from transformers import AutoTokenizer
from main import CrossEncoderModel
from ts.torch_handler.base_handler import BaseHandler
from faker import Faker
import random
from transformers import BertTokenizer
from transformers import AutoModel

class CrossEncoderHandler(BaseHandler): 
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.document_list = []
        K = 500
        fake = Faker()
        for i in range(K):
            ### random 300 words from wikipedia
            self.document_list.append(" ".join(fake.words(nb=300)))

    def initialize(self, ctx):
        # Load model and tokenizer
        self.model_dir = ctx.system_properties.get("model_dir")

        # Print GPU information
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print(f"Current GPU index: {torch.cuda.current_device()}")
        print(f"GPU name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        self.model = CrossEncoderModel("bert-base-uncased")
        self.model.load_state_dict(torch.load(f"{self.model_dir}/cross_encoder_model.pt", map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def preprocess(self, data):
        # Convert input text to tensor
        input_text = data[0]["body"]

        ### random 40 documents from the document list
        documents = random.sample(self.document_list, 400)

        query_document_pairs = [f"{input_text} {document}" for document in documents]
        inputs = self.tokenizer(query_document_pairs, return_tensors="pt", padding=True, truncation=True)
        return inputs.to(self.device)

    def inference(self, inputs):
        # Perform inference
        with torch.no_grad():
            logits = self.model(inputs)
        logits = logits.reshape(1, -1)
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        return probabilities.tolist()

    def postprocess(self, inference_output):
        # Return the inference result
        torch.cuda.empty_cache()    
        return inference_output
    

if __name__ == "__main__":
    from types import SimpleNamespace
    
    handler = CrossEncoderHandler() # type: ignore
    ctx = SimpleNamespace(**{
        "system_properties": {
            "model_dir": "cross_encoder_model"
        }
    })
    handler.initialize(ctx)
    data = [{"body": "What is the capital of France?"}]
    inputs = handler.preprocess(data)
    outputs = handler.inference(inputs)
    outputs = handler.postprocess(outputs)
    print(outputs)
