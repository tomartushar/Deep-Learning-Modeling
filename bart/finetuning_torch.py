import torch
import torch.nn as nn
import pickle as pk
import numpy as np
from datasets import load_dataset
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import AdamW, get_linear_schedule_with_warmup
import torch.optim as optim
from tqdm import tqdm

def get_data(saved = False):
    if saved:
      dataset = pk.load(open('data.pk', 'rb'))
    else:
        dataset = load_dataset("wmt14","de-en")
        pk.dump(dataset, open('data.pk', 'wb'))

    train_split = dataset['train'].select(range(30))
    val_split = dataset['validation'].select(range(10))

    train_input = [ex['en'] for ex in train_split['translation']]
    train_target = [ex['de'] for ex in train_split['translation']]
    val_input = [ex['en'] for ex in val_split['translation']]
    val_target = [ex['de'] for ex in val_split['translation']]

    return train_input, train_target, val_input, val_target


def prepare_data(tokenizer, inputs, targets, max_length=20):
    inputs = tokenizer.batch_encode_plus(inputs, max_length=max_length, padding=True, truncation=True, return_tensors='pt')
    targets = tokenizer.batch_encode_plus(targets, max_length=max_length, padding=True, truncation=True, return_tensors='pt')

    return inputs['input_ids'], inputs['attention_mask'], targets['input_ids']

def prepare_dataloader(data, batch_size = 2, sequential = False):
    dataset = TensorDataset(*data)
    sampler = SequentialSampler(dataset) if sequential else RandomSampler(dataset)
    return DataLoader(dataset, sampler = sampler, batch_size=batch_size)

def train(model, tokenizer, train_dataloader, val_dataloader, epochs = 2):

    optimizer = AdamW(model.parameters(), lr=1e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=-1)

    for epoch in range(epochs):
        train_loss = 0.0
        model.train()
        bar = tqdm(train_dataloader)
        for batch in bar:
            optimizer.zero_grad()
            input_ids, attention_mask, labels = batch
            labels[labels == tokenizer.pad_token_id] = -100
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            loss.backward()
            optimizer.step()
            scheduler.step()
            bar.set_description(f'Epoch {epoch}')
            bar.set_postfix({'loss': loss.item()})
            train_loss += loss.item()
        
        print(f"Train loss: {train_loss/len(train_dataloader)}")

        model.eval()
        val_loss = 0.0
        for batch in val_dataloader:
            input_ids, attention_mask, labels = batch
            labels[labels == tokenizer.pad_token_id] = -100
            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            val_loss += loss
        print(f"Val loss: {val_loss/len(val_dataloader)}")


def adv_train(model, tokenizer, train_dataloader, val_dataloader, 
              gradient_accumulation_steps = 6, epochs = 5, eval_steps = 10):

    optimizer = optim.AdamW(model.parameters(), lr=1e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=-1)

    n_steps = 0

    for epoch in range(epochs):
        print(f"** Epoch {epoch+1} **")
        loss = 0.0
        model.train()
        bar = tqdm(train_dataloader)
        for batch in bar:
            input_ids, attention_mask, labels = batch
            labels[labels == tokenizer.pad_token_id] = -100
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            
            unit_loss = outputs[0]*len(batch)/gradient_accumulation_steps
            unit_loss.backward()
            loss += unit_loss.detach()
            
            n_steps += len(batch)
            if n_steps % gradient_accumulation_steps == 0:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                bar.set_description(f"loss: {loss.item()}")
                loss = 0 
                optimizer.zero_grad()
                model.zero_grad()

            if n_steps%eval_steps != 0:
                continue
            
            model.eval()
            print('Examples Output:')
            ip = tokenizer.batch_decode(input_ids, skip_special_tokens=True)[0]
            labels[labels==-100] = tokenizer.pad_token_id
            tgt = tokenizer.batch_decode(labels, skip_special_tokens=True)[0]
            with torch.no_grad():
                op = tokenizer.batch_decode(model.generate(input_ids=input_ids, \
                            attention_mask=attention_mask, return_dict_in_generate=False, \
                            num_beams=8, num_return_sequences=1, max_length=512), \
                            skip_special_tokens=True)[0]
            print(f'Input: {ip}')
            print(f'Target: {tgt}')
            print(f'Output: {op}')
            
            n_val_steps = 0
            val_loss = 0.0
            val_bar = tqdm(val_dataloader)
            for batch in val_bar:
                input_ids, attention_mask, labels = batch
                labels[labels == tokenizer.pad_token_id] = -100
                with torch.no_grad():
                    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs[0]
                val_loss += loss
                n_val_steps += len(batch)
            
            print(f"Val loss: {val_loss/n_val_steps}")
            model.train()


def test1(texts):
    inputs = tokenizer(texts, return_tensors="pt")
    outputs = model.generate(**inputs) # inputs is dictionary containing `input_ids` and `attention_mask` 

    print(f"Output Tokens: {outputs}")
    print(f"Output sentence: {tokenizer.batch_decode(outputs, skip_special_tokens=True)}")

def test2(texts):
    inputs = tokenizer(texts, return_tensors="pt")
    outputs = model(**inputs)
    preds = np.argmax(outputs.logits.detach().numpy(), axis = -1)

    print(f"Output Tokens: {preds}")
    print(f"Output sentences: {tokenizer.batch_decode(preds, skip_special_tokens=True)}")



if __name__=='__main__':
    train_input, train_target, val_input, val_target = get_data(saved = True)

    tokenizer = AutoTokenizer.from_pretrained('facebook/bart-base')
    train_dataloader = prepare_dataloader(prepare_data(tokenizer, train_input, train_target))
    val_dataloader = prepare_dataloader(prepare_data(tokenizer, val_input, val_target), sequential=True)

    model = AutoModelForSeq2SeqLM.from_pretrained('facebook/bart-base')
    # train(model, tokenizer, train_dataloader, val_dataloader)
    adv_train(model, tokenizer, train_dataloader, val_dataloader)

    text = ["This is a long sentence that needs to be summarized."]
    test1(text)
    test2(text)


