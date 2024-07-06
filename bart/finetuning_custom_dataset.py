import torch
import numpy as np
import pickle as pk

from torch.utils.data import Dataset
from transformers import BartForConditionalGeneration, BartTokenizer
from transformers import TrainingArguments, Trainer

MODEL_NAME = "facebook/bart-base"


# Loading dataset
input_train = ['Resumption of the session',
               'I declare resumed the session of the European Parliament adjourned on Friday 17 December 1999, and I would like once again to wish you a happy new year in the hope that you enjoyed a pleasant festive period.',
               "Although, as you will have seen, the dreaded 'millennium bug' failed to materialise, still the people in a number of countries suffered a series of natural disasters that truly were dreadful.",
               'You have requested a debate on this subject in the course of the next few days, during this part-session.']
input_val = ['A Republican strategy to counter the re-election of Obama',
             'Republican leaders justified their policy by the need to combat electoral fraud.']
target_train = ['Wiederaufnahme der Sitzungsperiode',
                'Ich erkläre die am Freitag, dem 17. Dezember unterbrochene Sitzungsperiode des Europäischen Parlaments für wiederaufgenommen, wünsche Ihnen nochmals alles Gute zum Jahreswechsel und hoffe, daß Sie schöne Ferien hatten.',
                'Wie Sie feststellen konnten, ist der gefürchtete "Millenium-Bug " nicht eingetreten. Doch sind Bürger einiger unserer Mitgliedstaaten Opfer von schrecklichen Naturkatastrophen geworden.',
                'Im Parlament besteht der Wunsch nach einer Aussprache im Verlauf dieser Sitzungsperiode in den nächsten Tagen.']
target_val = ['Eine republikanische Strategie, um der Wiederwahl von Obama entgegenzutreten',
              'Die Führungskräfte der Republikaner rechtfertigen ihre Politik mit der Notwendigkeit, den Wahlbetrug zu bekämpfen.']



# Load tokenizer and model
def get_tokenizer(model_name=MODEL_NAME):
    return BartTokenizer.from_pretrained(model_name)

def get_model(model_name=MODEL_NAME):
    return BartForConditionalGeneration.from_pretrained(model_name)

tokenizer, model = get_tokenizer(), get_model()


# prepare dataset
class MyDataset(Dataset):
  def __init__(self, input_dataset, target_dataset, tokenizer, max_length=20):
    self.input_dataset = input_dataset
    self.target_dataset = target_dataset
    self.tokenizer = tokenizer
    self.max_length = max_length

  def __len__(self):
    return len(self.input_dataset)
  
  def __getitem__(self, idx):
    inputs = self.tokenizer(self.input_dataset[idx], max_length=self.max_length, 
                               truncation=True, padding='max_length', return_tensors='pt')
    with self.tokenizer.as_target_tokenizer():
      targets = self.tokenizer(self.target_dataset[idx], max_length=self.max_length, 
                               truncation=True, padding='max_length', return_tensors='pt')
    
    return {
        'input_ids': inputs['input_ids'][0],
        'attention_mask': inputs['attention_mask'][0],
        'labels': targets['input_ids'][0]
    }

train_dataset = MyDataset(input_train, target_train, tokenizer)
val_dataset = MyDataset(input_val, target_val, tokenizer)


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


loss_fn = torch.nn.CrossEntropyLoss(ignore_index = 1)

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
