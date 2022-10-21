import pickle
path = '/prot/lkz/LSTM/results/roi_events/region-001_events_new/21styear.pkl'
file=open(path, "rb")
a = pickle.load(file)
print(a)
print(len(a))
