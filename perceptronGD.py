import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_classification


class PerceptronGD:
    """Perceptron classifier with Batch Gradient Descent.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    random_state : int
      Random number generator seed for random weight initialization.

    Attributes
    -----------
    w_ : 1d-array
      Weights after fitting.
    b_ : Scalar
      Bias unit after fitting.
    errors_ : list
      Number of misclassifications in each epoch.
    """
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
          Training vectors.
        y : array-like, shape = [n_examples]
          Target values.

        Returns
        -------
        self : object
        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.float64(0.)
        
        self.errors_ = []

        for _ in range(self.n_iter):
            # 1. คำนวณค่าเน็ตอินพุตและทำนายผลสำหรับข้อมูลทั้งหมดพร้อมกัน
            net_output = self.net_input(X)
            output = np.where(net_output >= 0.0, 1, 0)
            
            # 2. หาค่าความผิดพลาด (Error) ของทั้งชุดข้อมูล
            errors = (y - output)
            
            # 3. อัปเดตค่าน้ำหนักและไบแอสครั้งเดียวต่อ 1 Epoch ด้วย Vectorization (X.T.dot)
            self.w_ += self.eta * X.T.dot(errors)
            self.b_ += self.eta * errors.sum()
            
            # 4. บันทึกจำนวนตัวที่ทายผิดในรอบนี้
            self.errors_.append(int(np.sum(errors != 0)))
            
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_) + self.b_

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, 0)

def plot_decision_regions(X, y, classifier, resolution=0.02):
    markers = ('o', 's', '^', 'v', '<')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # สร้าง Mesh Grid สำหรับพล็อตพื้นที่
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    
    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    lab = lab.reshape(xx1.shape)
    
    plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # พล็อตจุดข้อมูลจริงลงไป
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.8, 
                    c=colors[idx],
                    marker=markers[idx], 
                    label=f'Class {cl}', 
                    edgecolor='black')


# --- 3. รันการเทรนและแสดงกราฟ ---
if __name__ == "__main__":
    # สร้างข้อมูลจำลอง 2 มิติที่สามารถแยกแยะได้ด้วยเส้นตรง (Linearly Separable)
    X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, 
                               n_informative=2, random_state=42, n_clusters_per_class=1)

    print(f"X: {X}")
    print(f"y: {y}")

    # เทรนโมเดล Batch Gradient Descent
    classifier = PerceptronGD(eta=0.01, n_iter=15, random_state=42)
    classifier.fit(X, y)
     # 3. แสดงผลจำนวนความผิดพลาดในแต่ละรอบ (Epoch)
    print("จำนวนความผิดพลาดในแต่ละรอบ (Errors):", classifier.errors_)
    print("ค่าน้ำหนักสุดท้าย (Weights):", classifier.w_)
    print("ค่าไบแอสสุดท้าย (Bias):", classifier.b_)
    
    # 4. ทดสอบทำนายข้อมูลใหม่
    X_test = np.array([[2.5, 3.1], [-1.2, -2.5],[1.5,-5.3],[-2.3,4.0],[1.0,2.0],[5.0,-5.0]]) #ฟีเจอร์ 2 ติดลบ = 0
    predictions = classifier.predict(X_test)
    print("ผลการทำนายข้อมูลใหม่:", predictions)
    # จัดหน้าจอแสดงผล 2 กราฟคู่กัน
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # กราฟที่ 1: Error History
    plt.sca(ax1)
    plt.plot(range(1, len(classifier.errors_) + 1), classifier.errors_, marker='o', color='b', lw=2)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Number of Misclassifications', fontsize=12)
    plt.title('Error History (Batch Gradient Descent)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)

    # กราฟที่ 2: Decision Region
    plt.sca(ax2)
    plot_decision_regions(X, y, classifier=classifier)
    plt.xlabel('Feature 1', fontsize=12)
    plt.ylabel('Feature 2', fontsize=12)
    plt.legend(loc='upper left', fontsize=10)
    plt.title('Perceptron GD - Decision Regions', fontsize=14)

    plt.tight_layout()
    plt.show()