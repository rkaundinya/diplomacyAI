def to_gpu(x):
  if torch.cuda.is_available():
    return x.to('cuda')
  
  return x.to('cpu')