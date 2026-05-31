import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv("IMDb Movies India.csv", encoding="latin1")

df = df[['Genre','Director','Actor 1','Actor 2','Actor 3','Rating']]

df.dropna(inplace=True)

le = LabelEncoder()

for col in ['Genre','Director','Actor 1','Actor 2','Actor 3']:
    df[col] = le.fit_transform(df[col])

X = df.drop("Rating", axis=1)
y = df["Rating"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))