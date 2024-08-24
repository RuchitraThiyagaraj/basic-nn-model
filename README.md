# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

To create a neural network model for predicting a continuous value from your dataset, start by designing the model to learn from input features and predict the target value. Train the model using a portion of your data to teach it how to make predictions. Validate its performance by testing it on a separate data set to ensure it’s learning effectively and not just memorizing the training data. Then, evaluate the model on new, unseen data to confirm its generalization capabilities. Focus on optimizing performance metrics such as Mean Squared Error or Mean Absolute Error to improve prediction accuracy. Finally, use the model to uncover patterns and trends in the data, which will help in making informed decisions and understanding the behavior of the target variable.

## Neural Network Model

![image](https://github.com/user-attachments/assets/424c6992-7fec-4532-b28c-a6c664234619)

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name:T.Ruchitra
### Register Number: 212223110043
```python

from google.colab import auth
import gspread
from google.auth import default
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

auth.authenticate_user()
creds,_ = default()
gc = gspread.authorize(creds)
worksheet = gc.open('exp 1').sheet1
rows = worksheet.get_all_values()

df = pd.DataFrame(rows[1:], columns=rows[0])
df = df.astype({'INPUT':'float'})
df = df.astype({'OUTPUT':'float'})

X = df[['INPUT']].values
y = df[['OUTPUT']].values

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.33,random_state = 33)
Scaler = MinMaxScaler()
Scaler.fit(X_train)
X_train1 = Scaler.transform(X_train)

AI_Brain = Sequential([
    Dense(units = 6, activation = 'relu', input_shape=[1]),
    Dense(units = 5, activation = 'relu'),
    Dense(units = 1)
])

AI_Brain.compile(optimizer= 'rmsprop', loss="mse")
AI_Brain.fit(X_train1,y_train,epochs=5000)
AI_Brain.summary()


loss_df = pd.DataFrame(AI_Brain.history.history)

loss_df.plot()

X_test1 = Scaler.transform(X_test)

AI_Brain.evaluate(X_test1,y_test)

X_n1 = [[20]]
X_n1_1 = Scaler.transform(X_n1)
AI_Brain.predict(X_n1_1)

```
## Dataset Information

![image](https://github.com/user-attachments/assets/b89ad533-adb0-4381-853e-bc35fd020971)

## OUTPUT

### Training Loss Vs Iteration Plot

![image](https://github.com/user-attachments/assets/92a4f061-d7ef-488c-98ec-f185d35a01ef)

### Test Data Root Mean Squared Error

![image](https://github.com/user-attachments/assets/1da45613-df82-4f15-a8de-2557f65207f6)

### New Sample Data Prediction

![image](https://github.com/user-attachments/assets/c4b18950-3f62-4808-8315-76c8139fbb41)

## RESULT

A neural network regression model for the given dataset is created sucessfully.
