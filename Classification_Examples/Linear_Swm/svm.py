from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.inspection import DecisionBoundaryDisplay

cancer = load_breast_cancer()
X = cancer.data[:, :2]
y = cancer.target

svm = SVC(kernel='linear', C=1)
svm.fit(X, y)

DecisionBoundaryDisplay.from_estimator(
        svm,
        X,
        response_method="predict",
        alpha=0.8,
        cmap="Pastel1",
        xlabel=cancer.feature_names[0],
        ylabel=cancer.feature_names[1],
)

plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
plt.title('Decision Boundary of SVM')
plt.savefig('decision_boundary.png')
plt.show()




