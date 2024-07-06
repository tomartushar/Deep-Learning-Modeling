import torch
import numpy as np
import pickle as pk

from datasets import load_dataset
from transformers import BartForConditionalGeneration, BartTokenizer
from transformers import TrainingArguments, Trainer

MODEL_NAME = "facebook/bart-base"

# Loading dataset
def get_data(saved = False):
    if not saved:
        dataset = load_dataset("wmt14", 'de-en')
        pk.dump(dataset, open('/content/data.pk', 'wb'))
    else:
        dataset = pk.load(open('data.pk', 'rb'))
    train_dataset = dataset['train'].select(range(30))
    val_dataset = dataset['validation'].select(range(10))
    return train_dataset, val_dataset

train_dataset, val_dataset = get_data(saved = True)
print(f"Train Dataset Obejct:{train_dataset}")
print(f"Train dataset examples: {train_dataset[0]}")


# Load tokenizer and model
def get_tokenizer(model_name=MODEL_NAME):
    return BartTokenizer.from_pretrained(model_name)

def get_model(model_name=MODEL_NAME):
    return BartForConditionalGeneration.from_pretrained(model_name)

tokenizer, model = get_tokenizer(), get_model()

# prepare dataset
def preprocess_function(examples):
    inputs = [ex['en'] for ex in examples['translation']]
    targets = [ex['de'] for ex in examples['translation']]
    model_inputs = tokenizer(inputs, max_length=20, truncation=True, padding='max_length', return_tensors='pt')
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=20, truncation=True, padding='max_length', return_tensors="pt")

    model_inputs['labels'] = labels['input_ids']
    return model_inputs

train_dataset = train_dataset.map(preprocess_function, batched=True)
val_dataset = val_dataset.map(preprocess_function, batched=True)


# Setting training arguments
training_args = TrainingArguments(
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    output_dir='./results',
    evaluation_strategy='epoch',
    logging_dir='./logs',
    logging_steps=10,
    num_train_epochs=3,
    save_steps=10,
    save_total_limit=2
)


loss_fn = torch.nn.CrossEntropyLoss(ignore_index = 1) # padding_token_index = 1

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    logits = torch.tensor(logits[0]).transpose(1,2)
    labels = torch.tensor(labels)
    
    loss = loss_fn(logits, labels)
    
    return {"categorical_cross_entropy_loss": loss.item()}


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

print('Training starts ...')
trainer.train()

print('Evaluation starts ...')
trainer.evaluate()


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

text = ["This is a long sentence that needs to be summarized."]
test1(text)
test2(text)
