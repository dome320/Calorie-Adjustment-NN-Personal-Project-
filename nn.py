import pandas as pd 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import numpy
from sklearn.model_selection import train_test_split


# Create a model class that inherits nn.Module 
class Model(nn.Module):
    # Input Layer (3 features of the week) -->
    # Hidden Layer 1(number of neurons) 
    # --> H2(n) -->
    # ouput (7 classes of caloric adjustments) 

    def __init__(self, in_features=3, h1=8, h2=9, out_features=7):
        super().__init__() 
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2) 
        self.out = nn.Linear(h2, out_features)

    def forward(self, x): 
        x = F.relu(self.fc1(x)) 
        x = F.relu(self.fc2(x)) 
        x = self.out(x)

        return x
    
# pick a manual seed for randomization 
torch.manual_seed(41) 

model = Model() 
#Load saved Model
model.load_state_dict(torch.load("calorie_nn.pt"))
model.eval()




#Training
"""
my_df = pd.read_csv("bulk_cut_dataset.csv")

#Train, Test, Split
adjustment_to_class = {
    -300: 0,
    -200: 1,
    -100: 2,
     0:   3,
     100: 4,
     200: 5,
     300: 6
}

X = my_df.drop('delta_cal_adjustment', axis=1)
y = my_df['delta_cal_adjustment'].map(adjustment_to_class)

# Convert these to numpy arrays 
X = X.values
y = y.values

# Convert to numpy
X = X.astype(float)

# Manually scale calories (column 0) because of 
X[:, 0] = X[:, 0] / 1000.0

X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.2,random_state=41)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test  = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.long)
y_test  = torch.tensor(y_test, dtype=torch.long)

#Set the criterion of model to measure the error, how far off the predictions are from the data 
criterion = nn.CrossEntropyLoss() 

#Choose Adam Omptimizer, lr = learning rate(if error doesn't go down as we learn after a bunch of iterations we probably want to lower our learning rate) 
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

#Train the Model
#Epochs? (one run through all of the training data in network)
epochs = 100 
losses = [] 

for i in range(epochs):
    #Go Foward and get a predicition 
    y_pred = model.forward(X_train) #Get predicted results 

    #Measure the loss or error, gonna be high at first
    loss = criterion(y_pred, y_train) #Predicted value vs y training value 

    #Keep Track of losses 
    losses.append(loss.item())

    #Print every 10 epochs 
    if i % 10 == 0: 
        print(f'Epoch: {i} and loss: {loss}')
    
    #Back Prop
    optimizer.zero_grad()
    loss.backward() 
    optimizer.step() 



#Evaluate model on test set 

with torch.no_grad(): #Turns off back propogation 
    y_eval = model.forward(X_test) #X_test are features from our test set and y_eval will be predictions 
    loss = criterion(y_eval, y_test) #find the loss or error 

correct = 0

with torch.no_grad(): 
    for i, data in enumerate(X_test):
        y_val = model.forward(data) 

        #What class our network thinks it is? 
        print(f'{i+1}.) {str(y_val)} \t {y_test[i]}')

        #Correct or not? 
        if y_val.argmax().item() == y_test[i].item():
            correct += 1


print(f'We got {correct} correct!')

"""

# Interactive prediction
class_to_adjustment = {v: k for k, v in adjustment_to_class.items()}

model.eval()
with torch.no_grad():
    while True:
        print("\nEnter weekly inputs (or type 'q' to quit).")

        raw = input("cal_avg_last_week (e.g., 2900): ").strip()
        if raw.lower() in ("q", "quit", "exit"):
            break
        cal_avg_last_week = float(raw)

        raw = input("bw_change_last_week in lbs (e.g., -0.7): ").strip()
        if raw.lower() in ("q", "quit", "exit"):
            break
        bw_change_last_week = float(raw)

        raw = input("target_rate_lbs_per_week (e.g., -0.5, 0.0, 0.5): ").strip()
        if raw.lower() in ("q", "quit", "exit"):
            break
        target_rate_lbs_per_week = float(raw)

        # Apply same scaling as training
        x = torch.tensor(
            [[cal_avg_last_week / 1000.0, bw_change_last_week, target_rate_lbs_per_week]],
            dtype=torch.float32
        )

        logits = model(x)                         # shape: (1, 7)
        pred_class = logits.argmax(dim=1).item()   # 0..6
        pred_adjustment = class_to_adjustment[pred_class]

        print(f"\nPredicted calorie adjustment: {pred_adjustment:+d} kcal/day")
