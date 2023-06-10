import torch
from EmbNet import EmbNet
from vocab import Vocab
from tqdm import tqdm
import numpy as np

exec(open('utilities.py').read())

data = np.load("../../data/NPY_Files/NERAnnotatedTradePrompts.npy", allow_pickle=True)

PATH = './NERClassifierNet.pth'

def convertString(sentence):
  strList = []
  res = [i for j in sentence.split() for i in (j, ' ')][:-1]
  for i in res:
    if "." in i:
      strList.extend(i.split("."))
      strList[-1] = '.'
    elif "?" in i:
      strList.extend(i.split("?"))
      strList[-1] = '?'
    elif "!" in i:
      strList.extend(i.split("!"))
      strList[-1] = '!'
    else:
      strList.append(i)

  strList.append('"')
  strList.insert(0, '"')
  return np.array([strList])

def convertText(sample, vocab):
  out = []
  for r in tqdm(sample):
    out.append(_sample2window(r, vocab))
  return out

def _sample2window(sample, vocab):
    _x = None
    windowSize = 2
    for ix in range(len(sample)):
        win = []
        for off in range(-windowSize, windowSize + 1):
            if 0 <= ix + off < len(sample):
                win.append(sample[ix + off].lower())
            else:
                #make a null string which we'll embed as zero vector
                win.append('')

        x = win2v(win, vocab)
        if _x == None:
          _x = x
        else:
          _x = torch.cat((_x, x),0)
    return _x

def win2v(win, vocab):
    return torch.LongTensor([vocab.encode(word) for word in win])
  
#load word embeddings
pretrainedEmbeddings = torch.load('../../data/NPY_Files/embeddings.pt')

embed_net = to_gpu(EmbNet(pretrainedEmbeddings))
embed_net.load_state_dict(torch.load(PATH))

embed_net.eval()
embed_net.zero_grad()

#Reload vocab
vocab = Vocab()
vocab.train(data[:,0])

t = convertString("I'm interested in trading 9 wheat for 11 aluminum. What do you think?")
valTest = convertText(t, vocab)
valTest = torch.reshape(valTest[0], (-1,5))
logits = embed_net(to_gpu(valTest))
preds = torch.argmax(logits, dim=-1)

print(preds)

t = convertString("Give me 5 horses for 3 gold.")
valTest = convertText(t, vocab)
valTest = torch.reshape(valTest[0], (-1,5))
logits = embed_net(to_gpu(valTest))
preds = torch.argmax(logits, dim=-1)

print(preds)

preds = preds.cpu()

#Check to make sure no repeat entities were found
vals,entityCounts = np.unique(preds[np.where(preds != 0)[0]], return_counts=True)
repeatEntitiesFound = len(np.where(entityCounts != 1)[0]) != 0

if (repeatEntitiesFound):
    print("Could not properly understand input")

t = t.flatten()
item1Amt = t[np.where(preds == 1)[0]]
item1 = t[np.where(preds == 2)[0]]
item2Amt = t[np.where(preds == 3)[0]]
item2 = t[np.where(preds == 4)[0]]