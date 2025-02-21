import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
import functools
from IPython import display as ipythondisplay
from tqdm import tqdm
from scipy.io.wavfile import write
import mitdeeplearning as mdl


if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("using mps device")
else:
    device = torch.device("cpu")
    print("mps device not found, using cpu")


songs = mdl.lab1.load_training_data()
example_song = songs[0]
print(example_song)

songs_joined = "\n\n".join(songs)
vocab = sorted(set(songs_joined))

char2idx = {u: i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)


def vectorize_string(string):
    vectorized_output = np.array([char2idx[char] for char in string])
    return vectorized_output

vectorized_songs = vectorize_string(songs_joined)

def get_batch(vectorized_songs, seq_length, batch_size):
    n = vectorized_songs.shape[0] - 1
    idx = np.random.choice(n - seq_length, batch_size)

    input_batch = [vectorized_songs[i: i + seq_length] for i in idx]
    output_batch = [vectorized_songs[i + 1: i + seq_length + 1] for i in idx]

    x_batch = torch.tensor(np.array(input_batch), dtype=torch.long)
    y_batch = torch.tensor(np.array(output_batch), dtype=torch.long)
    return x_batch, y_batch


test_args = (vectorized_songs, 10, 2)
x_batch, y_batch = get_batch(*test_args)

### rnn lstm model

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        # layer 1: embedding layer to transform indices into dense vectors of a fixed size
        self.embdedding = nn.Embedding(vocab_size, embedding_dim)

        # layer 2: lstm with input = embedding dim, output = hidden size
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True)

        # layer 3: linear layer with input = hidden size and output = vocab size
        self.fc = nn.Linear(hidden_size, vocab_size)

    def init_hidden(self, batch_size, device):
        # initialize hidden state and cell state with 0s
        return (torch.zeros(1, batch_size, self.hidden_size).to(device),
                torch.zeros(1, batch_size, self.hidden_size).to(device))

    def forward(self, x, state=None, return_state=False):
        x = self.embdedding(x)

        if state is None:
            state = self.init_hidden(x.size(0), x.device)
        out, state = self.lstm(x, state)

        out = self.fc(out)
        return out if not return_state else (out, state)



vocab_size = len(vocab)
embdedding_dim = 256
hidden_size = 1024
batch_size = 8

model = LSTMModel(vocab_size, embdedding_dim, hidden_size).to(device)
# print(model)

x, y = get_batch(vectorized_songs, seq_length=100, batch_size=32)
x = x.to(device)
y = y.to(device)

pred = model(x)
# print("Input shape:      ", x.shape, " # (batch_size, sequence_length)")
# print("Prediction shape: ", pred.shape, "# (batch_size, sequence_length, vocab_size)")

sampled_indices = torch.multinomial(torch.softmax(pred[0], dim=-1), num_samples=1)
sampled_indices = sampled_indices.squeeze(-1).cpu().numpy()
sampled_indices

#
# print("Input: \n", repr("".join(idx2char[x[0].cpu()])))
# print()
# print("Next Char Predictions: \n", repr("".join(idx2char[sampled_indices])))

cross_entropy = nn.CrossEntropyLoss()
def compute_loss(labels, logits):
    batched_labels = labels.view(-1)
    batched_logits = logits.view(-1, logits.size(-1))
    loss = cross_entropy(batched_logits, batched_labels)
    return loss

y.shape
pred.shape
example_batch_loss = compute_loss(y, pred)

print(f"prediction shape: {pred.shape} # (batch_size, sequence_length, vocab_size)")
print(f"scalar_loss:      {example_batch_loss.mean().item()}")

vocab_size = len(vocab)

params = dict(
    num_training_iterations = 3000,
    batch_size = 8,
    seq_length = 100,
    learning_rate = 5e-3,
    embeddding_dim = 256,
    hidden_size = 1024,
)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "my_ckpt")
os.makedirs(checkpoint_dir, exist_ok=True)

model = LSTMModel(vocab_size, params["embeddding_dim"], params["hidden_size"])
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
def train_step(x, y):
    model.train()
    optimizer.zero_grad()

    y_hat = model(x)
    loss = compute_loss(y, y_hat)
    loss.backward()
    optimizer.step()
    return loss

history = []
plotter = mdl.util.PeriodicPlotter(sec=2, xlabel='iterations', ylabel='loss')
for iter in tqdm(range(params['num_training_iterations'])):
    x_batch, y_batch = get_batch(vectorized_songs, seq_length=params["seq_length"], batch_size=params["batch_size"])
    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)

    loss = train_step(x_batch, y_batch)
    history.append(loss.item())

    if iter % 100 == 0:
        print(f"iteration {iter}, loss: {loss.item():.4f}")
        plotter.plot(history)

    if iter % 500 == 0:
        torch.save({
            'epoch': iter,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, checkpoint_prefix)

print("training complete")

def generate_text(model, start_string, generation_length=1000):
    input_idx = [char2idx[s] for s in start_string]
    input_idx = torch.tensor([input_idx], dtype=torch.long).to(device)

    state = model.init_hidden(input_idx.size(0), device)

    text_generated = []

    for i in tqdm(range(generation_length)):
        predictions, state = model(input_idx, state, return_state=True)
        predictions = predictions.squeeze(0)
        input_idx = torch.multinomial(torch.softmax(predictions, dim=-1), num_samples=1)
        text_generated.append(idx2char[input_idx].item())

    return (start_string + ''.join(text_generated))


generated_text = generate_text(model, start_string="X", generation_length=1000)

generated_songs = mdl.lab1.extract_song_snippet(generated_text)

for i, song in enumerate(generated_songs):
    waveform = mdl.lab1.play_song(song)

    if waveform:
        print("Generated song", i)
        ipythondisplay.display(waveform)

        numeric_data = np.frombuffer(waveform.data, dtype=np.int16)
        wav_file_path = f"output_{i}.wav"
        write(wav_file_path, 88200, numeric_data)
