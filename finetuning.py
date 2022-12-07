# Fine-tune poetry generation model using csv file containing individual lines of poetry.
# Output is saved model and model weights to local machine under the name "finetuned_model.pt"
# and "finetuned_model_weights.pt".

# Author: Shravya Nandyala snandyal@andrew.cmu.edu

# Referenced https://github.com/dontmindifiduda/poe/blob/master/gpt2_poem_line.ipynb as a tutorial for fine-tuning.

import nltk
nltk.download('stopwords')
nltk.download('punkt')
import numpy as np
import pandas as pd 
import random
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
plt.style.use('bmh')
import torch
from transformers import GPT2LMHeadModel, GPT2Config, AdamW, get_linear_schedule_with_warmup
from py.generative_poetry.experiments.rugpt_with_stress.stressed_gpt_tokenizer import StressedGptTokenizer
from torch.utils.data import Dataset, random_split, DataLoader, RandomSampler, SequentialSampler

RANDOM_SEED = 73
MAX_SEQUENCE_LENGTH = 512

# Custom tokenizer from our base model
# Base model is from https://github.com/Koziev/verslibre
tokenizer = StressedGptTokenizer.from_pretrained('./model')

# Add special tokens for fine-tuning
special_tokens_dict = {'bos_token': '', 'eos_token': '', 'pad_token': '[PAD]'}
num_added_tokens = tokenizer.add_special_tokens(special_tokens_dict)

class PoemDataset(Dataset):
    def __init__(self, data, tokenizer, gpt2_type='gpt2', max_length=MAX_SEQUENCE_LENGTH):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.attn_masks = []
        
        for i in data:
            encodings_dict = tokenizer('' + i + '', truncation=True, max_length=max_length, padding='max_length')
            self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
            self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx]
  
#Train/Validation Split
def train_val_split(split, dataset):
    train_size = int(split * len(dataset))
    val_size = len(dataset) - train_size
    return train_size, val_size
     
# Instantiate DataLoaders and Define Model Creation Function
def create_dataloaders(train_dataset, val_dataset, bs):
    train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=bs)
    val_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=bs)
    
    return train_dataloader, val_dataloader
     
def finetune(train_dataloader, val_dataloader, config, epochs, lr, eps, warmup):
    model = GPT2LMHeadModel.from_pretrained('./model', config=config)
    model.resize_token_embeddings(len(tokenizer))
    optimizer = AdamW(model.parameters(), lr=lr, eps=eps)

    total_steps = len(train_dataloader) * epochs
    print(f'{total_steps} steps per epoch')
    scheduler = get_linear_schedule_with_warmup(optimizer,cnum_warmup_steps=warmup, num_training_steps=total_steps)

    for e in range(0, epochs):
        print(f'Epoch {e + 1} of {epochs}')
        total_train_loss = 0
        model.train()

        for step, batch in enumerate(train_dataloader):
            b_input_ids = batch[0]
            b_labels = batch[0]
            b_masks = batch[1]

            model.zero_grad()  
            outputs = model(b_input_ids, labels=b_labels, attention_mask=b_masks, token_type_ids=None)
            loss = outputs[0]  

            batch_loss = loss.item()
            total_train_loss += batch_loss

            loss.backward()
            optimizer.step()
            scheduler.step()
            print(f"Step {step} done")

        avg_train_loss = total_train_loss / len(train_dataloader)       

        print(f'Average Training Loss: {avg_train_loss}')
        print('Evaluating Model')

        model.eval()
        total_eval_loss = 0

        for batch in val_dataloader:
            b_input_ids = batch[0]
            b_labels = batch[0]
            b_masks = batch[1]

            with torch.no_grad():        
                outputs  = model(b_input_ids, attention_mask=b_masks, labels=b_labels)
                loss = outputs[0]  

            batch_loss = loss.item()
            total_eval_loss += batch_loss        

        avg_val_loss = total_eval_loss / len(val_dataloader)
        print(f'Validation loss: {avg_val_loss}')
    return model
 
def main():
    # Load lines of poetry from pushkin-poems.csv
    poem_line_df = pd.read_csv("pushkin-poems.csv")
    poem_line_df = poem_line_df.fillna('')

    # pushkin-poems.csv file has column "text" for each line
    poem_line_dataset = PoemDataset(poem_line_df['text'].values, tokenizer, max_length=MAX_SEQUENCE_LENGTH)
    train_size, val_size = train_val_split(0.8, poem_line_dataset)
    train_data, val_data = random_split(poem_line_dataset, [train_size, val_size])
   
    # Apply Random Seeds
    torch.cuda.manual_seed_all(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    # Set hyperparameters
    batch_size = 4
    learning_rate = 5e-4
    eps = 1e-8
    warmup_steps = 1e2

    train_dataloader, val_dataloader = create_dataloaders(train_data, val_data, batch_size)
    configuration = GPT2Config(vocab_size=len(tokenizer), n_positions=MAX_SEQUENCE_LENGTH).from_pretrained('./model', output_hidden_states=True)

    # Fine-tune model
    model = finetune(train_dataloader, val_dataloader, configuration, learning_rate, eps, warmup_steps)

    # Save model to computer
    torch.save(model, './finetuned_model.pt') 
    torch.save(model.state_dict(), './finetuned_model_state.pt')

if __name__ == "__main__":
    main()