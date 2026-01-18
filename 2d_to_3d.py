import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 이미지 로드
image = cv2.imread('sample.jpg')
if image is None:
    print("이미지를 찾을 수 없습니다.")
    exit()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 데이터 가공
gray_small = cv2.resize(gray, (100, 100)) 
h, w = gray_small.shape
X, Y = np.meshgrid(np.arange(w), np.arange(h))
Z = gray_small.astype(np.float32)

# 시각화 설정
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# 포인트 클라우드
ax.scatter(X, Y, Z, c=Z.flatten(), cmap='jet', s=1) 

ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Depth (Z)')
plt.title('Real 3D Point Cloud')
plt.show()