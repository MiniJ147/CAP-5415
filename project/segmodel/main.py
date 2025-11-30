from models import DDRNet_23 as DDRNet
import pickle
import pprint

model = DDRNet.get_seg_model(False)

with open("102.pkl", "rb") as f:
    data = pickle.load(f) 
pprint.pprint(data,width=120)