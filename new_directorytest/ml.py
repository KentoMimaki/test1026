import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# 1. データの準備
# 例としてXとyのデータをランダム生成します
np.random.seed(0)
X = 2 * np.random.rand(100, 1)  # 説明変数（特徴量）
y = 4 + 3 * X + np.random.randn(100, 1)  # 目的変数（出力）

# 2. データをトレーニングとテストに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 3. モデルの構築と学習
model = LinearRegression()
model.fit(X_train, y_train)

# 4. 予測
y_pred = model.predict(X_test)

# 5. 結果の表示
print("回帰係数（傾き）:", model.coef_)
print("切片:", model.intercept_)
print("決定係数 R^2:", model.score(X_test, y_test))

# 6. グラフ描画
plt.scatter(X, y, color='blue', label='データ')
plt.plot(X_test, y_pred, color='red', label='予測直線')
plt.xlabel("X")
plt.ylabel("y")
plt.title("単回帰分析")
plt.legend()
plt.show()
