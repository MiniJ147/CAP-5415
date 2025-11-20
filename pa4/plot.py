import matplotlib.pyplot as plt
import numpy as np
# IDC about hard coding the input XXXXX
conv3 = """Epoch-Iter: 1 30
Train set: Average loss: 2.1966, Accuracy: 13097/50000 (26%)

Test set: Average loss: 2.0716, Accuracy: 3826/10000 (38%)

Epoch-Iter: 2 30
Train set: Average loss: 2.0524, Accuracy: 20193/50000 (40%)

Test set: Average loss: 1.9963, Accuracy: 4624/10000 (46%)

Epoch-Iter: 3 30
Train set: Average loss: 1.9956, Accuracy: 23134/50000 (46%)

Test set: Average loss: 1.9906, Accuracy: 4651/10000 (47%)

Epoch-Iter: 4 30
Train set: Average loss: 1.9498, Accuracy: 25428/50000 (51%)

Test set: Average loss: 1.9228, Accuracy: 5358/10000 (54%)

Epoch-Iter: 5 30
Train set: Average loss: 1.9105, Accuracy: 27454/50000 (55%)

Test set: Average loss: 1.9081, Accuracy: 5505/10000 (55%)

Epoch-Iter: 6 30
Train set: Average loss: 1.8781, Accuracy: 29074/50000 (58%)

Test set: Average loss: 1.8832, Accuracy: 5742/10000 (57%)

Epoch-Iter: 7 30
Train set: Average loss: 1.8438, Accuracy: 30857/50000 (62%)

Test set: Average loss: 1.8756, Accuracy: 5833/10000 (58%)

Epoch-Iter: 8 30
Train set: Average loss: 1.8123, Accuracy: 32432/50000 (65%)

Test set: Average loss: 1.8364, Accuracy: 6229/10000 (62%)

Epoch-Iter: 9 30
Train set: Average loss: 1.7827, Accuracy: 33950/50000 (68%)

Test set: Average loss: 1.8087, Accuracy: 6494/10000 (65%)

Epoch-Iter: 10 30
Train set: Average loss: 1.7576, Accuracy: 35255/50000 (71%)

Test set: Average loss: 1.8100, Accuracy: 6519/10000 (65%)

Epoch-Iter: 11 30
Train set: Average loss: 1.7372, Accuracy: 36216/50000 (72%)

Test set: Average loss: 1.7937, Accuracy: 6649/10000 (66%)

Epoch-Iter: 12 30
Train set: Average loss: 1.7131, Accuracy: 37461/50000 (75%)

Test set: Average loss: 1.7807, Accuracy: 6800/10000 (68%)

Epoch-Iter: 13 30
Train set: Average loss: 1.6908, Accuracy: 38569/50000 (77%)

Test set: Average loss: 1.7687, Accuracy: 6904/10000 (69%)

Epoch-Iter: 14 30
Train set: Average loss: 1.6732, Accuracy: 39494/50000 (79%)

Test set: Average loss: 1.7708, Accuracy: 6893/10000 (69%)

Epoch-Iter: 15 30
Train set: Average loss: 1.6540, Accuracy: 40448/50000 (81%)

Test set: Average loss: 1.7676, Accuracy: 6919/10000 (69%)

Epoch-Iter: 16 30
Train set: Average loss: 1.6381, Accuracy: 41271/50000 (83%)

Test set: Average loss: 1.7725, Accuracy: 6862/10000 (69%)

Epoch-Iter: 17 30
Train set: Average loss: 1.6252, Accuracy: 41900/50000 (84%)

Test set: Average loss: 1.7675, Accuracy: 6937/10000 (69%)

Epoch-Iter: 18 30
Train set: Average loss: 1.6105, Accuracy: 42624/50000 (85%)

Test set: Average loss: 1.7718, Accuracy: 6878/10000 (69%)

Epoch-Iter: 19 30
Train set: Average loss: 1.5990, Accuracy: 43207/50000 (86%)

Test set: Average loss: 1.7594, Accuracy: 6981/10000 (70%)

Epoch-Iter: 20 30
Train set: Average loss: 1.5869, Accuracy: 43800/50000 (88%)

Test set: Average loss: 1.7492, Accuracy: 7106/10000 (71%)

Epoch-Iter: 21 30
Train set: Average loss: 1.5778, Accuracy: 44247/50000 (88%)

Test set: Average loss: 1.7451, Accuracy: 7146/10000 (71%)

Epoch-Iter: 22 30
Train set: Average loss: 1.5691, Accuracy: 44686/50000 (89%)

Test set: Average loss: 1.7495, Accuracy: 7109/10000 (71%)

Epoch-Iter: 23 30
Train set: Average loss: 1.5644, Accuracy: 44916/50000 (90%)

Test set: Average loss: 1.7485, Accuracy: 7102/10000 (71%)

Epoch-Iter: 24 30
Train set: Average loss: 1.5588, Accuracy: 45163/50000 (90%)

Test set: Average loss: 1.7490, Accuracy: 7109/10000 (71%)

Epoch-Iter: 25 30
Train set: Average loss: 1.5520, Accuracy: 45521/50000 (91%)

Test set: Average loss: 1.7571, Accuracy: 7034/10000 (70%)

Epoch-Iter: 26 30
Train set: Average loss: 1.5476, Accuracy: 45741/50000 (91%)

Test set: Average loss: 1.7462, Accuracy: 7135/10000 (71%)

Epoch-Iter: 27 30
Train set: Average loss: 1.5444, Accuracy: 45873/50000 (92%)

Test set: Average loss: 1.7445, Accuracy: 7132/10000 (71%)

Epoch-Iter: 28 30
Train set: Average loss: 1.5418, Accuracy: 46014/50000 (92%)

Test set: Average loss: 1.7438, Accuracy: 7146/10000 (71%)

Epoch-Iter: 29 30
Train set: Average loss: 1.5387, Accuracy: 46159/50000 (92%)

Test set: Average loss: 1.7412, Accuracy: 7178/10000 (72%)

Epoch-Iter: 30 30
Train set: Average loss: 1.5350, Accuracy: 46356/50000 (93%)

Test set: Average loss: 1.7360, Accuracy: 7237/10000 (72%)"""
conv5 = """Epoch-Iter: 1 30
Train set: Average loss: 2.1269, Accuracy: 16435/50000 (33%)

Test set: Average loss: 1.9851, Accuracy: 4752/10000 (48%)

Epoch-Iter: 2 30
Train set: Average loss: 1.9555, Accuracy: 25177/50000 (50%)

Test set: Average loss: 1.8830, Accuracy: 5786/10000 (58%)

Epoch-Iter: 3 30
Train set: Average loss: 1.8721, Accuracy: 29465/50000 (59%)

Test set: Average loss: 1.8197, Accuracy: 6436/10000 (64%)

Epoch-Iter: 4 30
Train set: Average loss: 1.8167, Accuracy: 32188/50000 (64%)

Test set: Average loss: 1.7907, Accuracy: 6704/10000 (67%)

Epoch-Iter: 5 30
Train set: Average loss: 1.7754, Accuracy: 34313/50000 (69%)

Test set: Average loss: 1.7919, Accuracy: 6691/10000 (67%)

Epoch-Iter: 6 30
Train set: Average loss: 1.7484, Accuracy: 35719/50000 (71%)

Test set: Average loss: 1.7529, Accuracy: 7080/10000 (71%)

Epoch-Iter: 7 30
Train set: Average loss: 1.7245, Accuracy: 36941/50000 (74%)

Test set: Average loss: 1.7263, Accuracy: 7352/10000 (74%)

Epoch-Iter: 8 30
Train set: Average loss: 1.7054, Accuracy: 37831/50000 (76%)

Test set: Average loss: 1.7154, Accuracy: 7465/10000 (75%)

Epoch-Iter: 9 30
Train set: Average loss: 1.6858, Accuracy: 38840/50000 (78%)

Test set: Average loss: 1.7042, Accuracy: 7565/10000 (76%)

Epoch-Iter: 10 30
Train set: Average loss: 1.6672, Accuracy: 39756/50000 (80%)

Test set: Average loss: 1.7006, Accuracy: 7609/10000 (76%)

Epoch-Iter: 11 30
Train set: Average loss: 1.6510, Accuracy: 40620/50000 (81%)

Test set: Average loss: 1.6819, Accuracy: 7799/10000 (78%)

Epoch-Iter: 12 30
Train set: Average loss: 1.6363, Accuracy: 41365/50000 (83%)

Test set: Average loss: 1.6740, Accuracy: 7859/10000 (79%)

Epoch-Iter: 13 30
Train set: Average loss: 1.6217, Accuracy: 42093/50000 (84%)

Test set: Average loss: 1.6689, Accuracy: 7915/10000 (79%)

Epoch-Iter: 14 30
Train set: Average loss: 1.6091, Accuracy: 42785/50000 (86%)

Test set: Average loss: 1.6618, Accuracy: 7994/10000 (80%)

Epoch-Iter: 15 30
Train set: Average loss: 1.5977, Accuracy: 43307/50000 (87%)

Test set: Average loss: 1.6752, Accuracy: 7875/10000 (79%)

Epoch-Iter: 16 30
Train set: Average loss: 1.5866, Accuracy: 43918/50000 (88%)

Test set: Average loss: 1.6578, Accuracy: 8041/10000 (80%)

Epoch-Iter: 17 30
Train set: Average loss: 1.5757, Accuracy: 44432/50000 (89%)

Test set: Average loss: 1.6507, Accuracy: 8107/10000 (81%)

Epoch-Iter: 18 30
Train set: Average loss: 1.5669, Accuracy: 44850/50000 (90%)

Test set: Average loss: 1.6505, Accuracy: 8096/10000 (81%)

Epoch-Iter: 19 30
Train set: Average loss: 1.5575, Accuracy: 45352/50000 (91%)

Test set: Average loss: 1.6494, Accuracy: 8108/10000 (81%)

Epoch-Iter: 20 30
Train set: Average loss: 1.5507, Accuracy: 45683/50000 (91%)

Test set: Average loss: 1.6388, Accuracy: 8233/10000 (82%)

Epoch-Iter: 21 30
Train set: Average loss: 1.5436, Accuracy: 46034/50000 (92%)

Test set: Average loss: 1.6550, Accuracy: 8055/10000 (81%)

Epoch-Iter: 22 30
Train set: Average loss: 1.5367, Accuracy: 46374/50000 (93%)

Test set: Average loss: 1.6398, Accuracy: 8197/10000 (82%)

Epoch-Iter: 23 30
Train set: Average loss: 1.5306, Accuracy: 46658/50000 (93%)

Test set: Average loss: 1.6429, Accuracy: 8178/10000 (82%)

Epoch-Iter: 24 30
Train set: Average loss: 1.5258, Accuracy: 46897/50000 (94%)

Test set: Average loss: 1.6353, Accuracy: 8267/10000 (83%)

Epoch-Iter: 25 30
Train set: Average loss: 1.5217, Accuracy: 47096/50000 (94%)

Test set: Average loss: 1.6366, Accuracy: 8240/10000 (82%)

Epoch-Iter: 26 30
Train set: Average loss: 1.5176, Accuracy: 47286/50000 (95%)

Test set: Average loss: 1.6323, Accuracy: 8288/10000 (83%)

Epoch-Iter: 27 30
Train set: Average loss: 1.5144, Accuracy: 47458/50000 (95%)

Test set: Average loss: 1.6456, Accuracy: 8148/10000 (81%)

Epoch-Iter: 28 30
Train set: Average loss: 1.5093, Accuracy: 47688/50000 (95%)

Test set: Average loss: 1.6286, Accuracy: 8318/10000 (83%)

Epoch-Iter: 29 30
Train set: Average loss: 1.5076, Accuracy: 47775/50000 (96%)

Test set: Average loss: 1.6278, Accuracy: 8345/10000 (83%)

Epoch-Iter: 30 30
Train set: Average loss: 1.5059, Accuracy: 47839/50000 (96%)

Test set: Average loss: 1.6345, Accuracy: 8264/10000 (83%)"""
conv7 = """Epoch-Iter: 1 30
Train set: Average loss: 2.1784, Accuracy: 13567/50000 (27%)

Test set: Average loss: 2.0365, Accuracy: 4216/10000 (42%)

Epoch-Iter: 2 30
Train set: Average loss: 1.9683, Accuracy: 24567/50000 (49%)

Test set: Average loss: 1.9408, Accuracy: 5198/10000 (52%)

Epoch-Iter: 3 30
Train set: Average loss: 1.8644, Accuracy: 29808/50000 (60%)

Test set: Average loss: 1.8273, Accuracy: 6307/10000 (63%)

Epoch-Iter: 4 30
Train set: Average loss: 1.8035, Accuracy: 32846/50000 (66%)

Test set: Average loss: 1.7832, Accuracy: 6788/10000 (68%)

Epoch-Iter: 5 30
Train set: Average loss: 1.7625, Accuracy: 34964/50000 (70%)

Test set: Average loss: 1.7500, Accuracy: 7111/10000 (71%)

Epoch-Iter: 6 30
Train set: Average loss: 1.7352, Accuracy: 36306/50000 (73%)

Test set: Average loss: 1.7352, Accuracy: 7253/10000 (73%)

Epoch-Iter: 7 30
Train set: Average loss: 1.7108, Accuracy: 37529/50000 (75%)

Test set: Average loss: 1.7199, Accuracy: 7414/10000 (74%)

Epoch-Iter: 8 30
Train set: Average loss: 1.6914, Accuracy: 38493/50000 (77%)

Test set: Average loss: 1.7029, Accuracy: 7569/10000 (76%)

Epoch-Iter: 9 30
Train set: Average loss: 1.6764, Accuracy: 39309/50000 (79%)

Test set: Average loss: 1.6759, Accuracy: 7868/10000 (79%)

Epoch-Iter: 10 30
Train set: Average loss: 1.6618, Accuracy: 39982/50000 (80%)

Test set: Average loss: 1.7064, Accuracy: 7530/10000 (75%)

Epoch-Iter: 11 30
Train set: Average loss: 1.6514, Accuracy: 40505/50000 (81%)

Test set: Average loss: 1.6557, Accuracy: 8058/10000 (81%)

Epoch-Iter: 12 30
Train set: Average loss: 1.6413, Accuracy: 41048/50000 (82%)

Test set: Average loss: 1.6546, Accuracy: 8067/10000 (81%)

Epoch-Iter: 13 30
Train set: Average loss: 1.6313, Accuracy: 41553/50000 (83%)

Test set: Average loss: 1.6712, Accuracy: 7893/10000 (79%)

Epoch-Iter: 14 30
Train set: Average loss: 1.6218, Accuracy: 41982/50000 (84%)

Test set: Average loss: 1.6392, Accuracy: 8205/10000 (82%)

Epoch-Iter: 15 30
Train set: Average loss: 1.6124, Accuracy: 42460/50000 (85%)

Test set: Average loss: 1.6453, Accuracy: 8162/10000 (82%)

Epoch-Iter: 16 30
Train set: Average loss: 1.6066, Accuracy: 42737/50000 (85%)

Test set: Average loss: 1.6361, Accuracy: 8245/10000 (82%)

Epoch-Iter: 17 30
Train set: Average loss: 1.5957, Accuracy: 43332/50000 (87%)

Test set: Average loss: 1.6294, Accuracy: 8320/10000 (83%)

Epoch-Iter: 18 30
Train set: Average loss: 1.5908, Accuracy: 43572/50000 (87%)

Test set: Average loss: 1.6300, Accuracy: 8303/10000 (83%)

Epoch-Iter: 19 30
Train set: Average loss: 1.5832, Accuracy: 43964/50000 (88%)

Test set: Average loss: 1.6213, Accuracy: 8404/10000 (84%)

Epoch-Iter: 20 30
Train set: Average loss: 1.5777, Accuracy: 44238/50000 (88%)

Test set: Average loss: 1.6283, Accuracy: 8320/10000 (83%)

Epoch-Iter: 21 30
Train set: Average loss: 1.5740, Accuracy: 44403/50000 (89%)

Test set: Average loss: 1.6243, Accuracy: 8381/10000 (84%)

Epoch-Iter: 22 30
Train set: Average loss: 1.5683, Accuracy: 44699/50000 (89%)

Test set: Average loss: 1.6182, Accuracy: 8423/10000 (84%)

Epoch-Iter: 23 30
Train set: Average loss: 1.5625, Accuracy: 44983/50000 (90%)

Test set: Average loss: 1.6175, Accuracy: 8432/10000 (84%)

Epoch-Iter: 24 30
Train set: Average loss: 1.5591, Accuracy: 45165/50000 (90%)

Test set: Average loss: 1.6134, Accuracy: 8466/10000 (85%)

Epoch-Iter: 25 30
Train set: Average loss: 1.5535, Accuracy: 45416/50000 (91%)

Test set: Average loss: 1.6182, Accuracy: 8435/10000 (84%)

Epoch-Iter: 26 30
Train set: Average loss: 1.5485, Accuracy: 45677/50000 (91%)

Test set: Average loss: 1.6053, Accuracy: 8552/10000 (86%)

Epoch-Iter: 27 30
Train set: Average loss: 1.5454, Accuracy: 45834/50000 (92%)

Test set: Average loss: 1.6080, Accuracy: 8517/10000 (85%)

Epoch-Iter: 28 30
Train set: Average loss: 1.5392, Accuracy: 46139/50000 (92%)

Test set: Average loss: 1.6155, Accuracy: 8445/10000 (84%)

Epoch-Iter: 29 30
Train set: Average loss: 1.5373, Accuracy: 46238/50000 (92%)

Test set: Average loss: 1.6081, Accuracy: 8517/10000 (85%)

Epoch-Iter: 30 30
Train set: Average loss: 1.5351, Accuracy: 46347/50000 (93%)

Test set: Average loss: 1.6092, Accuracy: 8527/10000 (85%)"""

def parse(input):
    """
    What is the grammar of our input?
    Epoch-Iter: Num 30
    Train set: ... (Num%)
    
    Test set: ... (Num%)
    """
    train_res = []
    test_res = []

    lines = input.split('\n')
    curr, mx = 0, 1e9
    while(curr < mx):
        iter, train, _, test = lines[:4]

        # edge case on the last line, since we have no ' ' 
        if len(lines) > 4:
            lines = lines[4:]
            lines.pop(0)

        # parse iter line to get number information 
        _, curr, mx = iter.split()
        curr, mx = int(curr), int(mx) 

        # parse train as str 
        def parse_accuracy(line):
            # grab last val [-1]
            # parse ( %) out of the str [1:-2]
            return int(line.split()[-1][1:-2]) 
        train_res.append(parse_accuracy(train))         
        test_res.append(parse_accuracy(test))

    return np.array(train_res), np.array(test_res)

# these are defined by the return from function above
TRAIN = 0
TEST = 1

domain = np.arange(start=1,stop=31)
conv3_accuracy = parse(conv3)
conv5_accuracy = parse(conv5)
conv7_accuracy = parse(conv7)

plot_info = [(TRAIN,"Train Accuracy"),(TEST,"Test Accuracy")]

# plot accuracy
for idx, title in plot_info: 
    plt.plot(domain, conv3_accuracy[idx], c="red", label="CNN 3 layers")
    plt.plot(domain, conv5_accuracy[idx], c="blue", label="CNN 5 layers")
    plt.plot(domain, conv7_accuracy[idx], c="green", label="CNN 7 layers")

    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.title(title)
    plt.legend()
    plt.savefig(title.replace(" ","-")+".png")
    plt.close()