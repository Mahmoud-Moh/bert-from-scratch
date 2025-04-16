
import os
import yaml
import torch
import torch.optim as optim
import sys
sys.path.append(os.path.join(os.path.dirname(__file__)), 'src')
sys.path.append(os.path.join(os.path.dirname(__file__)), 'data')
sys.path.append(os.path.join(os.path.dirname(__file__)), 'config')
import torch.nn as nn
from transformers import BertTokenizer
from model import BertEncoder
from util import *
from dataset import prepare_imdb_dataloaders
import math
import time

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config", "config.yaml")
def load_config(config_path=CONFIG_PATH):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

config = load_config(CONFIG_PATH)
BATCH_SIZE = int(config["training"]["batch_size"])
EPOCHS = int(config["training"]["epochs"])
MAX_LENGTH = int(config["training"]["max_length"])
SHUFFLE = bool(config["training"]["shuffle"])
D_MODEL = int(config["training"]["d_model"])
N_HEADS = int(config["training"]["n_heads"])
FFN_HIDDEN = int(config["training"]["ffn_hidden"])
N_LAYERS = int(config["training"]["n_layers"])
INIT_LR = float(config["training"]["init_lr"])
WEIGHT_DECAY = float(config["training"]["weight_decay"])
ADAM_EPS = float(config["training"]["adam_eps"])
LR_SCHEDULE_FACTOR = float(config["training"]["lr_scheduler"]["factor"])
LR_SCHEDULE_PATIENCE = int(config["training"]["lr_scheduler"]["patience"])
DATA_PERCENTAGE = float(config["training"]["data_percentage"])
VOCAB_SIZE = int(config["training"]["vocab_size"])
N_CLASSES = int(config["training"]["n_classes"])
PRETRAINED_MODEL_NAME = 'bert-base-uncased'


DEVICE =  "cuda" if torch.cuda.is_available() else "cpu"

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

#Tokenizer
tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)


#Model
model = BertEncoder(d_model=D_MODEL, n_heads=N_HEADS, ffn_hidden=FFN_HIDDEN, n_layers=N_LAYERS, max_len=MAX_LENGTH, vocab_size=VOCAB_SIZE, n_classes=N_CLASSES, device=DEVICE)

#Optimizer
optimizer = optim.Adam(params=model.parameters(), lr=INIT_LR, weight_decay=WEIGHT_DECAY)

#Loss fn
loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

#LR Scheduler
lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=LR_SCHEDULE_PATIENCE, factor=LR_SCHEDULE_FACTOR)

#Training Function
def train_one_epoch(model, optimizer, loss_fn, dl):
    epoch_loss = 0.0
    batches_eval_matrics = []
    for i, batch in enumerate(dl):
        src = batch['input_ids'].to(DEVICE) #Tensor of size [Batch Size, max_length]
        trg_class = batch['label'].to(DEVICE) #Integer tensor of size [1]
        #print("Src: ", src.shape)
        #print("Trg: ", src.shape)

        optimizer.zero_grad()

        predicted_class, logits = model(src)
        #print("Predicted_class", predicted_class.shape, predicted_class)
        #print("Trg_class", trg_class.shape, trg_class)
        loss = loss_fn(logits, trg_class)
        batch_eval_metrics = eval_metrics(loss.item(), trg_class.detach().cpu().numpy(), predicted_class.detach().cpu().numpy())
        batches_eval_matrics.append(batch_eval_metrics)

        loss.backward()
        optimizer.step()

        epoch_loss += loss
        print('step :', round((i / len(dl)) * 100, 2), '% , loss :', loss.item())
    return epoch_loss, batches_eval_matrics

def eval_one_epoch(model, loss_fn, dl):
    model.eval()
    epoch_loss = 0.0
    batches_eval_matrics = []
    with torch.no_grad():
        for i, batch in enumerate(dl):
            src = batch['input_ids'].to(DEVICE) #Tensor of size [Batch Size, max_length]
            trg_class = batch['label'].to(DEVICE) #Integer tensor of size [1]

            predicted_class, logits = model(src)

            loss = loss_fn(logits, trg_class)
            batch_eval_metrics = eval_metrics(loss.item(), trg_class.detach().cpu().numpy(), predicted_class.detach().cpu().numpy())
            batches_eval_matrics.append(batch_eval_metrics)

            epoch_loss += loss
            print('step :', round((i / len(dl)) * 100, 2), '% , loss :', loss.item())
    return epoch_loss, batches_eval_matrics


def evaluate(model, loss_fn, dl):
    model.eval()
    all_preds = []
    all_targets = []
    total_loss = 0.0
    with torch.no_grad():
        for batch in dl:
            src = batch['input_ids'].to(DEVICE)
            trg_class = batch['label'].to(DEVICE)

            predicted_class, logits = model(src)
            loss = loss_fn(logits, trg_class)
            total_loss += loss.item()

            all_preds.extend(predicted_class.cpu().numpy())
            all_targets.extend(trg_class.cpu().numpy())

    # Compute final metrics
    final_metrics = eval_metrics(
        batch_loss=total_loss / len(dl),
        batch_y_true=all_targets,
        batch_y_pred=all_preds
    )

    print("\n=== Test Set Evaluation ===")
    print(f"Loss     : {final_metrics['loss']:.4f}")
    print(f"Accuracy : {final_metrics['accuracy']:.4f}")
    print(f"Precision: {final_metrics['precision']:.4f}")
    print(f"Recall   : {final_metrics['recall']:.4f}")
    print(f"F1 Score : {final_metrics['f1']:.4f}")

    return final_metrics



def run(epochs, model, optimizer, loss_fn, train_dl, val_dl):
    train_losses, eval_losses = [], []
    train_epochs_eval_metrics, val_epochs_eval_metrics = [], []
    for epoch in range(epochs):
        start_time = time.time()
        print(f"start_time: {start_time}")
        train_epoch_loss, train_batches_eval_matrics = train_one_epoch(model, optimizer, loss_fn, train_dl)
        val_epoch_loss, val_batches_eval_matrics = eval_one_epoch(model, loss_fn, val_dl)
        end_time = time.time()
        train_epochs_eval_metrics.append(train_batches_eval_matrics)
        val_epochs_eval_metrics.append(val_batches_eval_matrics)

        lr_scheduler.step(val_epoch_loss)

        train_losses.append(train_epoch_loss)
        eval_losses.append(val_epoch_loss)
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        all_train_loss = [metrics['loss'] for batch_metrics in train_epochs_eval_metrics for metrics in batch_metrics]
        all_train_accuracy = [metrics['accuracy'] for batch_metrics in train_epochs_eval_metrics for metrics in batch_metrics]
        all_train_f1 = [metrics['f1'] for batch_metrics in train_epochs_eval_metrics for metrics in batch_metrics]

        all_val_loss = [metrics['loss'] for batch_metrics in val_epochs_eval_metrics for metrics in batch_metrics]
        all_val_accuracy = [metrics['accuracy'] for batch_metrics in val_epochs_eval_metrics for metrics in batch_metrics]
        all_val_f1 = [metrics['f1'] for batch_metrics in val_epochs_eval_metrics for metrics in batch_metrics]

        """
        f = open(f'result/{epoch}_train_loss.txt', 'w')
        f.write(str(train_losses))
        f.close()

        f = open(f'result/{epoch}_valid_loss.txt', 'w')
        f.write(str(eval_losses))
        f.close()
        """
        print(f'Epoch: {epoch + 1} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_epoch_loss:.3f} | Train PPL: {math.exp(train_epoch_loss):7.3f}')
        print(f'\tVal Loss: {val_epoch_loss:.3f} |  Val PPL: {math.exp(val_epoch_loss):7.3f}')
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    f = open(f'result/train_loss.txt', 'w')
    f.write(str(all_train_loss))
    f.close()

    f = open(f'result/valid_loss.txt', 'w')
    f.write(str(all_val_loss))
    f.close()

    f = open(f'result/train_acc.txt', 'w')
    f.write(str(all_train_accuracy))
    f.close()

    f = open(f'result/valid_acc.txt', 'w')
    f.write(str(all_val_accuracy))
    f.close()

    f = open(f'result/train_f1.txt', 'w')
    f.write(str(all_train_f1))
    f.close()

    f = open(f'result/{epoch}_valid_f1.txt', 'w')
    f.write(str(all_val_f1))
    f.close()



if __name__ == "__main__":
    data_path = "./data/1/movie.csv"
    train_dl, val_dl, test_dl = prepare_imdb_dataloaders(csv_path=data_path, tokenizer=tokenizer,
                                                         batch_size=BATCH_SIZE, max_length=MAX_LENGTH, data_percentage=DATA_PERCENTAGE)
    run(EPOCHS, model, optimizer, loss_fn, train_dl, val_dl)
    torch.save(model.state_dict(), 'bert-imdb50K.pth')
    evaluate(model, loss_fn, test_dl)
