import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.1, n_iterations=10):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape # n_samples=4 จำนวนแถว  n_features = 2 จำนวน colum
        # เริ่มต้นน้ำหนัก (Weights) และค่า Bias เป็นศูนย์
        self.weights = np.zeros(n_features) 
        print(self.weights)
        self.bias = 0

        for _ in range(self.n_iterations):
            for idx, x_i in enumerate(X):
                # คำนวณค่าลัพธ์เชิงเส้น: z = w*x + b
                linear_output = np.dot(x_i, self.weights) + self.bias
                # ใช้ Activation Function (Step Function)
                y_predicted = self.activation(linear_output)
                
                # ปรับปรุงค่าน้ำหนักและ Bias ตามความผิดพลาด
                update = self.learning_rate * (y[idx] - y_predicted)
                print(f"update : {update}")
                self.weights += update * x_i
                print(f"update weights: {self.weights}")
                self.bias += update

    def activation(self, x):

        activated_output = np.where(x >= 0, 1, 0) # if x >= 0 x=1 else x = 0
        print(f"Linear output (x) = {x} | Activated output = {activated_output}")
        return activated_output

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return self.activation(linear_output)

if __name__ == "__main__":
    # ข้อมูลจำลอง AND Gate
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
        ])
    y = np.array([0, 1, 1, 1])

    # สร้างและเทรนโมเดล
    p = Perceptron(learning_rate=0.1, n_iterations=10)
    p.fit(X, y)

    # ทดสอบการทำนาย
    predictions = p.predict(X)
    print("ผลลัพธ์การทำนาย:", predictions)
    print("Weights ที่ได้:", p.weights)
    print("Bias ที่ได้:", p.bias)