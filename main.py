import torch
import torch.nn as nn
from transformers import AutoModel
from transformers import AutoTokenizer
import argparse


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

class CrossEncoderModel(nn.Module):
    def __init__(self, bert_model_name):
        super(CrossEncoderModel, self).__init__()
        self.bert_model = self._load_bert_model(bert_model_name)
        self.projector = nn.Linear(self.bert_model.config.hidden_size, 1)

    def _load_bert_model(self, bert_model_name):
        model = AutoModel.from_pretrained(bert_model_name)
        # Freeze the model parameters to make it easier to TorchScript
        for param in model.parameters():
            param.requires_grad = False
        return model

    def forward(self, inputs):
        '''
        Instead of taking a dictionary, we now take explicit arguments
        '''
        # forward the tokenized query-document pairs to the model

        ### if batch_size is too large, we need to split the inputs into smaller batches (20 batch_size maximal)
        if inputs.input_ids.shape[0] > 20:
            print("Batch size is too large, splitting into smaller batches")
            for i in range(0, inputs.input_ids.shape[0], 20):
                batch_inputs = inputs.input_ids[i:i+20]
                outputs = self.bert_model(batch_inputs)
        else:
            outputs = self.bert_model(
                **inputs
            )
        # Get the [CLS] token representation (first token)
        cls_representation = outputs.last_hidden_state[:, 0, :]
        # project the [CLS] token representation to 1 dimension
        scores = self.projector(cls_representation)
        return scores

def test_model(model, tokenizer):
    query = "What is the capital of France?"
    documents = ["France is a country in Europe.", "Paris is the capital of France.", "France is known for its fashion industry."]

    # concatenate the query and documents 
    query_document_pairs = [query + " " + doc for doc in documents]

    inputs = tokenizer(query_document_pairs, padding=True, truncation=True, return_tensors="pt")
    
    # Pass individual tensors instead of dictionary
    outputs = model(inputs)
    print(outputs)

# model = SimpleModel()
# scripted_model = torch.jit.script(model)
# scripted_model.save("simple_model.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and save cross-encoder model')
    parser.add_argument('--test', action='store_true', help='Run test before saving')
    args = parser.parse_args()

    model = CrossEncoderModel("bert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


    # Test the model if --test flag is provided
    if args.test:
        print("Testing model...")
        test_model(model, tokenizer)

    # Save the model for TorchServe
    print("Saving model...")
    model_dir = "cross_encoder_model"
    tokenizer.save_pretrained(model_dir)
    torch.save(model.state_dict(), f"{model_dir}/cross_encoder_model.pt")


### cross-encoder model
# Handler:
### input query, input K documents (handler initialize the documents)
### tokenize the K query-document pairs 
### forward the tokenized query-document pairs to the model
### postprocess the model output to get the score for each document
### return the score for each document

# Model:
### input: tokenized query-document pairs
### output: score for each document
