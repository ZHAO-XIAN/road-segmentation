import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from skimage import measure

# 1. 載入圖片並轉為灰階
image = cv2.imread('test4.jpg')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 2. 計算 LBP 特徵
radius = 1  # LBP 半徑
n_points = 8 * radius  # LBP 邻域點數
lbp_image = local_binary_pattern(gray_image, n_points, radius, method='uniform')

# 3. 對 LBP 圖像進行閾值處理 (將其轉換為二值圖像)
lbp_binary = lbp_image > 0  # 可以調整閾值來分割特定紋理特徵

# 4. 進行形態學處理來獲得更乾淨的二值圖像
kernel = np.ones((3, 3), np.uint8)
cleaned_image = cv2.morphologyEx(lbp_binary.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

# 5. 構建 Watershed 所需的貯水點 (種子點)
# 可以根據二值圖像進行輪廓檢測來找出種子點
contours, _ = cv2.findContours(cleaned_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
marker = np.zeros_like(cleaned_image, dtype=np.int32)

# 將輪廓設為不同的標記
for i, contour in enumerate(contours):
    cv2.drawContours(marker, [contour], -1, i + 1, -1)

# 6. 進行 Watershed 分割
# 將標記為 -1 的部分設置為邊界（這是 Watershed 的要求）
marker = np.where(marker == 0, -1, marker)

# 使用 Watershed 算法來分割圖像
cv2.watershed(image, marker)

# 7. 繪製最終分割結果 (可以設定道路區域的顏色)
image[marker == -1] = [0, 0, 255]  # 用紅色標記邊界
image[marker > 0] = [0, 255, 0]  # 用綠色標記分割區域

# 顯示結果
cv2.imshow('Watershed Result', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
