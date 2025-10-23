# init 
source venv/bin/activate

# STEP 1 ================ #
echo "STEP 1"

python3 train_evaluate_CNN.py --mode 1 --num_epochs 60 --batch_size 10 --learning_rate 0.1

echo "DONE"
# ====================== #

# STEP 2
echo "STEP 2"

python3 train_evaluate_CNN.py --mode 2 --num_epochs 60 --batch_size 10 --learning_rate 0.1

echo "DONE"
# ====================== #

# STEP 3
echo "STEP 3"

python3 train_evaluate_CNN.py --mode 3 --num_epochs 60 --batch_size 10 --learning_rate 0.03

echo "DONE"
# ====================== #

# STEP 4
echo "STEP 4"

python3 train_evaluate_CNN.py --mode 4 --num_epochs 60 --batch_size 10 --learning_rate 0.03

echo "DONE"
# ====================== #

# STEP 5
echo "STEP 5"

python3 train_evaluate_CNN.py --mode 3 --num_epochs 40 --batch_size 10 --learning_rate 0.03

echo "DONE"
# ====================== #

echo "TERMINATE"