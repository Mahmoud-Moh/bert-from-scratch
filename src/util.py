def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
def eval_metrics(batch_loss, batch_y_true, batch_y_pred):
  accuracy_sc = accuracy_score(batch_y_true, batch_y_pred)
  precision_sc = precision_score(batch_y_true, batch_y_pred)
  recall_sc = recall_score(batch_y_true, batch_y_pred)
  f1_sc = f1_score(batch_y_true, batch_y_pred)
  return {
        'loss': batch_loss,
        'accuracy': accuracy_sc,
        'precision': precision_sc,
        'recall': recall_sc,
        'f1': f1_sc
    }
