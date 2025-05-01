import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score

data = pd.read_csv("/content/heart.csv")
X = data.drop("target", axis=1)
y = data["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree.fit(X_train, y_train)

plt.figure(figsize=(12, 8))
plot_tree(decision_tree, filled=True, feature_names=X.columns, class_names=["No Disease", "Disease"])
plt.title("Decision Tree")
plt.show()

train_acc = []
test_acc = []
max_depths = range(1, 11)

for depth in max_depths:
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    model.fit(X_train, y_train)
    train_acc.append(model.score(X_train, y_train))
    test_acc.append(model.score(X_test, y_test))

plt.plot(max_depths, train_acc, label='Train')
plt.plot(max_depths, test_acc, label='Test')
plt.xlabel("Max Depth")
plt.ylabel("Accuracy")
plt.title("Decision Tree Depth vs Accuracy")
plt.legend()
plt.show()

forest = RandomForestClassifier(n_estimators=100, random_state=42)
forest.fit(X_train, y_train)

tree_acc = accuracy_score(y_test, decision_tree.predict(X_test))
forest_acc = accuracy_score(y_test, forest.predict(X_test))

print("Decision Tree Accuracy:", tree_acc)
print("Random Forest Accuracy:", forest_acc)

feature_scores = forest.feature_importances_
feature_names = X.columns
sorted_features = np.argsort(feature_scores)[::-1]

print("\nFeature Importances:")
for i in sorted_features:
    print(f"{feature_names[i]}: {feature_scores[i]:.4f}")

plt.bar(range(len(feature_scores)), feature_scores[sorted_features])
plt.xticks(range(len(feature_scores)), [feature_names[i] for i in sorted_features], rotation=45)
plt.title("Random Forest Feature Importance")
plt.tight_layout()
plt.show()

tree_cv = cross_val_score(decision_tree, X, y, cv=5)
forest_cv = cross_val_score(forest, X, y, cv=5)

print("\nCross-Validation Accuracy:")
print("Decision Tree: %.2f ± %.2f" % (tree_cv.mean(), tree_cv.std()))
print("Random Forest: %.2f ± %.2f" % (forest_cv.mean(), forest_cv.std()))
