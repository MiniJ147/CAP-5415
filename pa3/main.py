# driver to run each file
# format: 
#   python3 main.py [type]
#   type => (encoder, classifier)
#   encoder ==> (cnn, mlp)
#   classifier ==> (knn)

from q1 import cnn_autoencoder
from q1 import mlp_autoencoder  
from q2 import knn_classifier

import sys

type_maps = {
    "cnn": cnn_autoencoder.main,
    "mlp": mlp_autoencoder.main,
    "knn": knn_classifier.main,
}

def fail_with_exit(title: str):
    print(title)
    print('\tpython3 main.py [type]\n\ttype => (encoder | classifier)\n\tencoder ==> (cnn | mlp)\n\tclassifier ==> knn')
    exit()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        fail_with_exit("incorrect number of args")

    type_idx = 1
    _type = sys.argv[type_idx]

    if not _type in type_maps:
        fail_with_exit("incorrect type")
    
    type_maps[_type]() # call the main function
