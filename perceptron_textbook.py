import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

class Perceptron:
    """Perceptron classifier.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    random_state : int
      Random number generator seed for random weight
      initialization.

    Attributes
    -----------
    w_ : 1d-array
      Weights after fitting.
    b_ : Scalar
      Bias unit after fitting.
    errors_ : list
      Number of misclassifications (updates) in each epoch.

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
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
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
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_ += update * xi
                self.b_ += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_) + self.b_

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, 0)


# สมมติว่ามีคลาส Perceptron อยู่ด้านบนแล้ว...

if __name__ == "__main__":
    # 1. สร้างชุดข้อมูลจำลอง (2 ฟีเจอร์, 2 คลาส)
    X, y = make_blobs(
        n_samples=100, 
        n_features=2, 
        centers=2, 
        cluster_std=0.50, 
        random_state=42
    )

    print(f"X: {X}")
    print(f"y:{y}")
    
    # หมายเหตุ: Perceptron ในโค้ดของคุณคาดหวังป้ายกำกับ (Label) เป็น 0 หรือ 1
    # แต่ make_blobs จะคืนค่าเป็น -1 กับ 1 หรือ 0 กับ 1 (ขึ้นอยู่กับการตั้งค่า)
    # ชุดข้อมูลนี้ค่า y จะเป็น 0 และ 1 พอดี เหมาะกับ Perceptron พอดีครับ

    # 2. สร้างและฝึกสอนโมเดล Perceptron
    pPN = Perceptron(eta=0.1, n_iter=10, random_state=42)
    pPN.fit(X, y)

    # 3. แสดงผลจำนวนความผิดพลาดในแต่ละรอบ (Epoch)
    print("จำนวนความผิดพลาดในแต่ละรอบ (Errors):", pPN.errors_)
    print("ค่าน้ำหนักสุดท้าย (Weights):", pPN.w_)
    print("ค่าไบแอสสุดท้าย (Bias):", pPN.b_)

    # 4. ทดสอบทำนายข้อมูลใหม่
    X_test = np.array([[2.5, 3.1], [-1.2, -2.5],[1.5,-5.3],[-2.3,4.0],[1.0,2.0],[5.0,-5.0]])
    predictions = pPN.predict(X_test)
    print("ผลการทำนายข้อมูลใหม่:", predictions)