import torch
import tiktoken
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import json
from torch.nn import functional as F

from gpt import GPT2
from fine_tuning import model_finetune, feedforward
from model_converter import load_from_standard_weights


# supports both Mac mps and CUDA
def compute_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'


            
# convert QA json formate to dataframe ['questions', 'answers']
def QA_json_to_dataframe(file_path):
    # read the json file
    file = json.loads(open(file_path).read())

    # parsing
    data = pd.json_normalize(file, ['data', 'paragraphs'])[['context', 'qas']]

    # List to store each row of the final DataFrame
    rows = []
    
    # Iterate through each row of the flattened DataFrame
    for index, row in data.iterrows():
        context = row['context']
        for qa_pair in row['qas']:
            question = qa_pair['question']
            answer = qa_pair['answers'][0]['text'] if qa_pair['answers'] else None
            rows.append([context, question, answer])
    
    # Create a DataFrame from the list of rows
    data = pd.DataFrame(rows, columns=['context', 'question', 'answer'])

    return data


    
# over writting the Dataset class to utilize binary file parsing
class QA_Dataset(Dataset):
    def __init__(self, dataframe, tokenizer, seq_length=1024):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        context, question, answer = item['context'], item['question'], item['answer']
        
        # Combine question, answer, and prompt template tokens
        combined_str = f'Context: {context} Question: {question} Answer: {answer}'

        # Tokenize the combined string
        combined_tokens = self.tokenizer.encode_ordinary(combined_str)
        
        # Exceeds max sequence length, Truncate 
        if len(combined_tokens) >= self.seq_length:
            combined_tokens = combined_tokens[-self.seq_length:]
        
        # pad the sequence to max_seq_length
        eot_tok = self.tokenizer._special_tokens['<|endoftext|>']
        combined_tokens += [eot_tok] * (self.seq_length-len(combined_tokens))
        
        # Create input-output pair (x, y)
        x = torch.tensor(combined_tokens[:-1])  # Remove last token
        y = torch.tensor(combined_tokens[1:])   # Shift by one

        return x, y
    



# Question and Answering inference
# inputs a question, generate an answer
@torch.no_grad()
def inference(model, tokenizer, context, question, temperature=1.0, top_k=None):
    model = model.eval()
    
    # Combine question, answer, and prompt template tokens
    prompt = f'Context: {context} Question: {question} Answer: '
    
    x = tokenizer.encode_ordinary(prompt) # text encoding
    input_len = len(x) # length of the prompt + question tokens
    x = torch.tensor(x, dtype=torch.long) # convert to tensor
    x = x.unsqueeze(0) # unsqueeze the batch dimension
    idx = x.to(compute_device()) # move to GPU device
    
    # inference
    max_new_tokens = 128 # give 128 tokens for the answer tokens
    
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):
        # crop idx to the last block_size tokens
        idx_cond = idx[:, -1024:]
       # forward the model to get the logits for the index in the sequence
        logits, _ = model(idx_cond)
        # pluck the logits at the final step and scale by desired temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop the logits to only the top k options
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
        # apply softmax to convert logits to (normalized) probabilities
        probs = F.softmax(logits, dim=-1)
        # sample from the distribution
        idx_next = torch.multinomial(probs, num_samples=1)
        # append sampled index to the running sequence and continue
        idx = torch.cat((idx, idx_next), dim=1)
        
    prediction = idx[0].tolist()
     
    # processing the prediction
    prediction = prediction[input_len:] # remove the questions tokens
    prediction = [token for token in prediction if token != tokenizer._special_tokens['<|endoftext|>']]# remove all <eos> tokens
    prediction = tokenizer.decode(prediction) # decoding to text
    prediction = prediction.strip() # strip the beginning/end newline and space character (if exists)
    return prediction
    

            
def main():
    # create GPT model
    num_embed = 768
    num_heads = 12
    num_layers = 12
    model = GPT2(num_embed, num_heads, num_layers)
    
    # load the pretained weights
    pretrained_path = '../pretrained_models/GPT/GPT2.bin'
    model.load_state_dict(load_from_standard_weights(pretrained_path))
    
    # use Low Rank Adaptation (LoRA) for fine tuning
    model.apply_lora() 
    
    # move model to GPU
    model = model.to(compute_device())
    
    # tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # loading question and answering datasets
    QA_path = '../Datasets/Stanford Question Answering Dataset/'
    df = QA_json_to_dataframe(QA_path + 'train-v1.1.json')
    
    # train-valid split
    train_df = df.iloc[:int(len(df)*0.9)]
    valid_df = df.iloc[int(len(df)*0.9):]
    
    # visualize some examples
    for i in range(0, len(train_df), len(train_df)//5):
        item = train_df.iloc[i]
        context, question, answer = item['context'], item['question'], item['answer']
        print('>>', context)
        print('>', question)
        print('=', answer)
        print()
    
    # create data loader
    # use seq_length=512 to speed up
    train_dataset = QA_Dataset(train_df, tokenizer, seq_length=512) 
    train_loader = DataLoader(train_dataset, batch_size=6, shuffle=True)
    valid_dataset = QA_Dataset(valid_df, tokenizer, seq_length=512) 
    valid_loader = DataLoader(valid_dataset, batch_size=12, shuffle=False)
    
    # fine tunning on question and answering tasks
    model_finetune(model, train_loader, valid_loader)
    
    # load fine tuned model
    model.load_state_dict(torch.load(f'{type(model).__name__}_finetuned.pth'))
    
    # test preformance
    test_df = QA_json_to_dataframe(QA_path + 'dev-v1.1.json')
    test_dataset = QA_Dataset(test_df, tokenizer, seq_length=512)
    test_loader = DataLoader(test_dataset, batch_size=12, shuffle=False)
    
    # randomly picked a few questions from validation dataset
    print('Test Performance')
    feedforward(model, test_loader)
    for i in range(0, len(test_df), len(test_df)//5):
        item = test_df.iloc[i]
        context, question, answer = item['context'], item['question'], item['answer']
        prediction = inference(model, tokenizer, context, question)
        print('>>', context)
        print('>', question)
        print('=', answer)
        print('<', prediction)
        print()
            
if __name__ == '__main__':
    main()