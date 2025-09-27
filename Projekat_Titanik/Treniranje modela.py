import os
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

# Ulazni fajl sa redukovanim podacima
IN = "processed_data_reduced.csv"

# Provera da li fajl postoji
if not os.path.exists(IN):
    raise FileNotFoundError(f"{IN} not found")

# 1. Učitavanje podataka
df = pd.read_csv(IN)

# 2. Podela na X (feature) i y (ciljna varijabla)
X = df.drop(columns=["Survived"])
y = df["Survived"]

# 3. Trening i test skup
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# -----------------------------------------------------------------
# MODELI BEZ OPTIMIZACIJE (default parametri)
# -----------------------------------------------------------------
base_models = {
    "Decision Tree (default)": DecisionTreeClassifier(random_state=0),
    "Random Forest (default)": RandomForestClassifier(random_state=0),
    "Gradient Boosting (default)": GradientBoostingClassifier(random_state=0),
    "XGBoost (default)": xgb.XGBClassifier(eval_metric='logloss', random_state=0),
    "Logistic Regression (default)": LogisticRegression(max_iter=5000, random_state=0)
}

print("\n=== REZULTATI MODELA BEZ OPTIMIZACIJE ===\n")
for name, model in base_models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"\n===== Classification Report za {name} =====\n")
    print(classification_report(y_test, y_pred))

# -----------------------------------------------------------------
#  MODELI SA OPTIMIZACIJOM (GridSearchCV)
# -----------------------------------------------------------------

# 4. GridSearch za Decision Tree
dt_params = {
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'class_weight': ['balanced']
}
dt_grid = GridSearchCV(
    estimator=DecisionTreeClassifier(random_state=0),
    param_grid=dt_params,
    scoring='accuracy',
    cv=5,
    n_jobs=-1,
    verbose=1
)
dt_grid.fit(X_train, y_train)
print("Najbolji parametri za Decision Tree:", dt_grid.best_params_, "\n")

# 4.1 GridSearch za Random Forest
rf_params = {
    'n_estimators': [100, 200],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'class_weight': ['balanced']
}
rf_grid = GridSearchCV(
    estimator=RandomForestClassifier(random_state=0),
    param_grid=rf_params,
    scoring='accuracy',
    cv=5,
    n_jobs=-1,
    verbose=1
)
rf_grid.fit(X_train, y_train)
print("Najbolji parametri za Random Forest:", rf_grid.best_params_, "\n")

# 4.2 GridSearch za Gradient Boosting
gb_params = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 5],
    'subsample': [0.8, 1.0]
}
gb_grid = GridSearchCV(
    estimator=GradientBoostingClassifier(random_state=0),
    param_grid=gb_params,
    scoring='accuracy',
    cv=5,
    n_jobs=-1,
    verbose=1
)
gb_grid.fit(X_train, y_train)
print("Najbolji parametri za Gradient Boosting:", gb_grid.best_params_, "\n")

# 4.3 GridSearch za XGBoost
xgb_params = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 5],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}
xgb_grid = GridSearchCV(
    estimator=xgb.XGBClassifier(eval_metric='logloss', random_state=0),
    param_grid=xgb_params,
    scoring='accuracy',
    cv=5,
    n_jobs=-1,
    verbose=1
)
xgb_grid.fit(X_train, y_train)
print("Najbolji parametri za XGBoost:", xgb_grid.best_params_, "\n")

# 4.4 GridSearch za Logistic Regression
lr_params = {
    'C': [0.01, 0.1, 1, 10],
    'penalty': ['l2'],
    'solver': ['lbfgs', 'saga'],
    'max_iter': [5000]
}
lr_grid = GridSearchCV(
    estimator=LogisticRegression(random_state=0),
    param_grid=lr_params,
    scoring='accuracy',
    cv=5,
    n_jobs=-1,
    verbose=1
)
lr_grid.fit(X_train, y_train)
print("Najbolji parametri za Logistic Regression:", lr_grid.best_params_, "\n")

# 5. Definišemo modele sa najboljim parametrima
models = {
    "Decision Tree": dt_grid.best_estimator_,
    "Random Forest": rf_grid.best_estimator_,
    "Gradient Boosting": gb_grid.best_estimator_,
    "XGBoost": xgb_grid.best_estimator_,
    "Logistic Regression": lr_grid.best_estimator_
}

# 6. Evaluacija optimizovanih modela
print("\n=== REZULTATI MODELA SA OPTIMIZACIJOM ===\n")
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"\n===== Classification Report za {name} =====\n")
    print(classification_report(y_test, y_pred))

# -----------------------------------------------------------------
# 7. Konfuzione matrice + Feature Importance za Decision Tree i Random Forest
# -----------------------------------------------------------------
for model_name in ["Decision Tree", "Random Forest"]:
    model = models[model_name]
    y_pred = model.predict(X_test)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Survived", "Survived"])
    disp.plot(cmap='Blues')
    plt.title(f"Confusion Matrix - {model_name}")
    plt.show()

    # Feature Importance
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        feature_names = X.columns
        feat_imp = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False)

        plt.figure(figsize=(8, 6))
        sns.barplot(x="Importance", y="Feature", data=feat_imp, color="skyblue")
        plt.title(f"Feature Importance - {model_name}")
        plt.tight_layout()
        plt.show()