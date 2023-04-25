import torch
import torch.nn as nn
import torch.optim as optim
import datasets
import gensim.downloader as api
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report

PATH = './NERClassifierNet.pth'
SHOULD_TRAIN = False
WORD_DIM = 50
NUM_CLASSES = 5

data = np.load("../../data/NPY_Files/NERAnnotatedTradePrompts.npy", allow_pickle=True)
model = api.load("glove-wiki-gigaword-50")

def to_gpu(x):
  if torch.cuda.is_available():
    return x.to('cuda')
  
  return x.to('cpu')

"""
Class that handles mapping to and from words
to vocab indices
"""
class Vocab:
    #stream is just a param for indexing into the token list of data
    def __init__(self, stream = 'tokens', target = False):
        self._target = target
        if self._target:
            self._word2idx = {}
        else:
            self._word2idx = {'__UNK__': 0, '':1}
        self._idx2word = {}
        self._stream = stream

    def train(self, raw_ds):
        #generate vocab list
        for sample in raw_ds:
            for token in sample:
                t = token.lower()
                if t not in self._word2idx:
                    self._word2idx[t] = len(self._word2idx)
        #rebuild reverse lookup
        self._idx2word = {v:k for k, v in self._word2idx.items()}

    def encode(self, word):
        if self._target:
            return self._word2idx[word]
        else:
            #Tries to get the word - if doesn't exist, returns '__UNK__'
            return self._word2idx.get(word, self._word2idx['__UNK__'])
    
    def decode(self, idx):
        if self._target:
            return self._idx2word[idx]
        else:
            #Tries to get the word - if doesn't exist, returns '__UNK__'
            return self._idx2word.get(idx, '__UNK__')
        
vocab = Vocab()
vocab.train(data[:,0])

class PyCoNLL2(torch.utils.data.Dataset):
    """
    A PyTorch Dataset wrapper for CoNLL2003 dataset

    Wrapper is instantiated by providing:
    * ds_split: specific split of CoNLL
    * vector_model: word vector model from gensim
    * window_size: window parameter for slicing (default: 2)
    """
    def __init__(self, ds_split, vocab, window_size=2):
        #cache params
        self._vocab = vocab
        self._raw = ds_split
        self._win = window_size

        #hard-coded but enumerated
        self._classes = NUM_CLASSES

        #setup data containers
        #will be a list of tensors
        self._x = []
        self._y = []

        #preprocess data
        for r in tqdm(self._raw):
            self._sample2window(r)

    def __len__(self):
        #length of dataset is simply number of samples
        return len(self._x)
    
    def __getitem__(self, item):
        #return both input and gold label when iterating
        return {
            'input': self._x[item],
            'label': self._y[item]
        }
    
    def _sample2window(self, sample):
        for ix in range(len(sample[0])):
            win = []
            for off in range(-self._win, self._win + 1):
                if 0 <= ix + off < len(sample[0]):
                    win.append(sample[0][ix + off].lower())
                else:
                    #make a null string which we'll embed as zero vector
                    win.append('')

            x = self._window2vector(win)
            y = sample[1][ix] # not one-hot anymore

            self._x.append(x)
            self._y.append(y)

    def _window2vector(self, win):
        return torch.LongTensor([self._vocab.encode(word) for word in win])
        
train = PyCoNLL2(data[0:2000,:], vocab)
val = PyCoNLL2(data[2000:3000,:], vocab)
#test = PyCoNLL2(ds['test'], vocab)

vocab_size = len(vocab._word2idx)
vocab_dim = 50
wvs = torch.zeros(vocab_size, vocab_dim)

for ix in range(vocab_size):
    word = vocab.decode(ix)
    if word in model:
        wvs[ix, :] = torch.tensor(model[word])
    else:
        #if we don't have word in pre-trained set, randomly initialize to a unique vec
        wvs[ix, :] = torch.randn(WORD_DIM)

class EmbNet(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, word_vectors):
        super(EmbNet, self).__init__()

        self._embed = nn.Embedding.from_pretrained(word_vectors, freeze=False)
        self._lin = nn.Linear(in_dim, hidden_dim)
        self._act = nn.Sigmoid()
        self._pred = nn.Linear(hidden_dim, out_dim, bias=False)

    #x is batchSize x windowLength
    def forward(self, x):
        #maps batchSize x windowLength -> batchSize x windowLength x wordDim
        emb = self._embed(x)

        #flattens dimensionality starting at 1st index: (batchSize x windowLength*wordDim)
        win_emb = torch.flatten(emb, start_dim=1)

        z = self._lin(win_emb)
        h = self._act(z)
        y_hat = self._pred(h)

        return y_hat
    
embed_net = to_gpu(EmbNet(5 * WORD_DIM, WORD_DIM, NUM_CLASSES, wvs))
opt = torch.optim.Adagrad(embed_net.parameters())
loss_fn = nn.CrossEntropyLoss(reduction='sum')

def train_early_stop(batch_size=128):
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=True)

    print(len(train_loader))

    epoch = 0
    training = True
    best_val_loss = None

    while training:
        print(f'Start epoch {epoch + 1}')

        #This call makes sure we're calculating gradients on our neural net params
        embed_net.train()

        train_loss = 0
        for xs in train_loader:
            #Clear the gradients before feed forward
            opt.zero_grad()

            #feed forward
            y_preds = embed_net(to_gpu(xs['input']))

            #calculate loss
            loss = loss_fn(y_preds, to_gpu(xs['label'].type(torch.LongTensor)))
            train_loss += loss.cpu().item()

            #Calculate gradients
            loss.backward()

            #update params
            opt.step()

        print(f'Current train loss: {train_loss:.2f}')

        #disable gradients during validation
        embed_net.eval()

        val_loss = 0
        for xs in val_loader:
            y_preds = embed_net(to_gpu(xs['input']))
            val_loss += loss_fn(y_preds, to_gpu(xs['label'].type(torch.LongTensor))).cpu().item()

        print(f'Total validation loss: {val_loss:.2f}')

        if not best_val_loss:
            best_val_loss = val_loss
        elif val_loss < best_val_loss:
            best_val_loss = val_loss
        else:
            print('No improvement - early stopping')
            training = False

        epoch += 1
        print()

if SHOULD_TRAIN:
    train_early_stop()
    torch.save(embed_net.state_dict(), PATH)
else:
    embed_net.load_state_dict(torch.load(PATH))

#Evaluate
embed_net.eval()

train_loader = torch.utils.data.DataLoader(train, batch_size=128, shuffle=True)
val_loader = torch.utils.data.DataLoader(val, batch_size=128, shuffle=True)

#train perf
gold = []
pred = []

for xs in train_loader:
    logits = embed_net(to_gpu(xs['input']))

    #take argmax over last dimension to get most probable class
    preds = torch.argmax(logits, dim=-1)

    #.cpu() to make sure we're on cpu before converting to numpy
    gold.extend(xs['label'].cpu().numpy())
    pred.extend(preds.cpu().numpy())

print(classification_report(gold, pred))

#val perf
gold = []
pred = []

for xs in val_loader:
    logits = embed_net(to_gpu(xs['input']))

    #take argmax over last dimension to get most probable class
    preds = torch.argmax(logits, dim=-1)

    #.cpu() to make sure we're on cpu before converting to numpy
    gold.extend(xs['label'].cpu().numpy())
    pred.extend(preds.cpu().numpy())

print(classification_report(gold, pred))