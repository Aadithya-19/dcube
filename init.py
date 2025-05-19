#initialization
#algorithm for error detection-basically i used ast for code parsing (syntax errors) and exec for runtime
#cuda for faster compilation, its gonna be very useful
#needs numba installation, and pandas installation (for some reason i didn't have pandas already installed on this system)

#needs Training Wheels protocol to be initialized

''' basic logic of the code
def analyze_code(code):
    if is_syntax_error(code):
        return "Syntax Error"
    elif is_runtime(code):
        return "Runtime Error"
    else:
        return "Logical Error"
def is_runtime(code):
    try:
        exec(code)
        return False
    except Exception:
        return True

def is_syntax_error(code):
    try:
        ast.parse(code)
        return False
    except SyntaxError:
        return True

for sample in dataset:
    print(f"Predicted: {analyze_code(sample['code'])}, Actual: {sample['label']}")
'''

import ast
from numba  import cuda
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer

#data
data = pd.read_csv("bugs_dataset.csv")
dataset = [
    #the dataset is going to be a bit different, for now I am using a basic distribution and how it is supposed to look.
    {"code": "print(5 + )", "label": "syntax"},
    {"code": "def divide(x, y): return x / y\nprint(divide(5, 0))", "label": "runtime"},
    {"code": "def is_even(n): return n % 2 == 1\nprint(is_even(2))", "label": "logical"}
]
df=pd.DataFrame(data) #dataset variable going to be replaced with the variable data, which will be reading a file filled with code


#Aadithya - analyze code and be able to distinguish between errors - very basic

#pre-processing
def preprocess_code(code):
    return code.strip().lower()

df['clean_code'] = df['code'].apply(preprocess_code)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['clean_code'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

def predict_error_type(code_snippet):
    processed = preprocess_code(code_snippet)
    features = vectorizer.transform([processed])
    return model.predict(features)[0]

#test sim
test_code = "while True print('infinite')"
print(predict_error_type(test_code))  # should output syntax


#Amogh - scoring here...

#Abhiraj - add the @cuda.jit part to the code and improve the vectorization algorithm i wrote
