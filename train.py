import json
from ntlk_utils import tokenize, stem, bag_of_words
import numpy as np
from model import NeuralNetwork

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

with open('intents.json', 'r') as file:
    intents = json.load(file)

# print(intents)

all_words = []
tags = []
xy = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)

    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

ignore_words = ['?', '!', '.', ',']
all_words = [stem(w) for w in all_words if w not in ignore_words]
# print(all_words)
# print(xy)

# take unique words
all_words = sorted(set(all_words))
tags = sorted(set(tags))
# print(xy)

x_train = []
y_train = []

for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    x_train.append(bag)

    label = tags.index(tag)
    y_train.append(label)  # crossEntropyLoss


x_train = np.array(x_train)
y_train = np.array(y_train)

# print(x_train)


class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(x_train)
        self.x_data = x_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples


# Hyperparameters
batch_size = 8
input_size = len(all_words)
hidden_size = 8
output_size = len(tags)
learning_rate = 0.001
num_epochs = 1000

dataset = ChatDataset()
train_loader = DataLoader(
    dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNetwork(input_size, hidden_size, output_size).to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop wrapped inside if __name__ == '__main__'
# to avoid multiprocessing issues on Windows with num_workers=2
if __name__ == '__main__':
    for epoch in range(num_epochs):
        for (words, labels) in train_loader:
            words = words.to(device)
            # labels = labels.to(device)
            labels = labels.to(device, dtype=torch.long)

            # forward
            outputs = model(words)
            loss = criterion(outputs, labels)

            # backword and optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch+1) % 10 == 0:
            print(f'epoch {epoch+1}/{num_epochs}, loss ={loss.item():.4f}')

    print(f'final loss, loss ={loss.item():.4f}')

    # Save/ load model

    data = {
        "model_state": model.state_dict(),
        "input_size": input_size,
        "output_size": output_size,
        "hidden_size": hidden_size,
        "all_words": all_words,
        "tags": tags
    }

    FILE = "data.pth"
    torch.save(data, FILE)

    print(f'training completed and files are saved in {FILE}')
