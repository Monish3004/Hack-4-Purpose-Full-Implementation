import pandas as pd
import plotly.express as px 
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import streamlit.components.v1 as components


from PIL import Image
   
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://img.freepik.com/free-vector/two-neon-red-hearts-background-with-text-space_1017-23294.jpg?size=626&ext=jpg&ga=GA1.1.638758127.1711705771&semt=ais");
background-size: 100%;
background-position: top left;
background-repeat: repeat;
background-attachment: local;
}}


[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}

[data-testid="stToolbar"] {{
right: 2rem;
}}
</style>
"""


st.set_page_config(page_title="Cupid's Choice ❤️❤️", page_icon=":heart:", layout="wide")
st.title(":heart: Cupid's Choice")

st.markdown(page_bg_img, unsafe_allow_html=True)


df = pd.read_excel(
        io="data.xlsx",
        engine="openpyxl",
        sheet_name="Sheet1",
        skiprows=0,
)
st.header("Love's Eternal Arrow 🏹💘")

# ---- MAINPAGE ----
st.header("Preferences:")
activity = st.selectbox(
    "Choose Preferred Date Activities:",
    options=df["Preferred Date Activities"].unique()
)

loc = st.selectbox(
    "Select the Location Preferences:",
    options=df["Location Preferences"].unique()
)

time = st.selectbox(
    "Select the Timing Preferences:",
    options=df["Timing Preferences"].unique()
)

budget = st.selectbox(
    "Select the Budget Preferences:",
    options=df["Budget Preferences"].unique())

# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_excel("data.xlsx")

# Encode categorical variables
label_encoders = {}
categorical_cols = ['Preferred Date Activities', 'Location Preferences', 'Timing Preferences', 'Budget Preferences']
for col in categorical_cols:
    label_encoders[col] = LabelEncoder()
    data[col] = label_encoders[col].fit_transform(data[col])

# Select input and output features
X = data[categorical_cols]
y = data['Venue ID']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the KNN model
k = 3  # Number of neighbors
knn_model = KNeighborsClassifier(n_neighbors=k)
knn_model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = knn_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Gather user input
user_input = {
    'Preferred Date Activities': activity,  # Example input, replace with actual user input
    'Location Preferences': loc,       # Example input, replace with actual user input
    'Timing Preferences': time,        # Example input, replace with actual user input
    'Budget Preferences': budget           # Example input, replace with actual user input
}

# Encode user input
encoded_input = {}
for col, val in user_input.items():
    encoded_input[col] = label_encoders[col].transform([val])[0]

# Convert encoded input to DataFrame
input_df = pd.DataFrame(encoded_input, index=[0])

# Make prediction
venue_id_prediction = knn_model.predict(input_df)

print("Predicted Venue ID:", venue_id_prediction[0])
st.write("Predicted Venue ID:", venue_id_prediction[0])
 

d = {1:"Wonderla",
     2:"Click Art Museum",
     3:"Besant Nagar",
     4:"Battlefield",
     5:"Zen Arts Academy",
     6:"Sporfy",
     7:"ECR & OMR",
     8:"Cooking or Baking with Movienight in Home",
     9:"Sathyam Cinemas",
     10:"Cream Story",
	11:"Pheonix Mall",
	12:"6th Avenue Restro Bar"}

st.write("Predicted Venue : ", d[venue_id_prediction[0]])
