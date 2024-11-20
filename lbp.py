import cv2
import numpy as np
import matplotlib.pyplot as plt

def calculate_lbp(image, radius=1, n_points=8):
    """
    使用NumPy向量化操作優化的LBP計算
    
    參數:
    image: 輸入的灰度圖像
    radius: 圓形鄰域的半徑
    n_points: 圓形鄰域上採樣點的數量
    
    返回:
    lbp_image: LBP特徵圖像
    """
    rows = image.shape[0]
    cols = image.shape[1]
    
    # 生成圓形鄰域的坐標
    angles = 2 * np.pi * np.arange(n_points) / n_points
    x = radius * np.cos(angles)
    y = -radius * np.sin(angles)
    
    # 獲取鄰域坐標的四個參考點（用於雙線性插值）
    x1 = np.floor(x).astype(int)
    x2 = np.ceil(x).astype(int)
    y1 = np.floor(y).astype(int)
    y2 = np.ceil(y).astype(int)
    
    # 計算插值權重
    fx = x - x1
    fy = y - y1
    
    # 準備插值權重矩陣
    w1 = (1 - fx) * (1 - fy)
    w2 = fx * (1 - fy)
    w3 = (1 - fx) * fy
    w4 = fx * fy
    
    # 初始化輸出圖像
    lbp_image = np.zeros((rows, cols), dtype=np.uint8)
    
    # 填充圖像以處理邊界
    padded_image = np.pad(image, ((radius, radius), (radius, radius)), 'edge')
    
    # 在中心區域計算LBP
    for i in range(n_points):
        # 計算四個參考點的鄰域值
        n1 = padded_image[radius+x1[i]:rows+radius+x1[i], 
                         radius+y1[i]:cols+radius+y1[i]]
        n2 = padded_image[radius+x2[i]:rows+radius+x2[i], 
                         radius+y1[i]:cols+radius+y1[i]]
        n3 = padded_image[radius+x1[i]:rows+radius+x1[i], 
                         radius+y2[i]:cols+radius+y2[i]]
        n4 = padded_image[radius+x2[i]:rows+radius+x2[i], 
                         radius+y2[i]:cols+radius+y2[i]]
        
        # 雙線性插值
        neighbor = w1[i]*n1 + w2[i]*n2 + w3[i]*n3 + w4[i]*n4
        
        # 比較中心像素和鄰域像素
        center = padded_image[radius:radius+rows, radius:radius+cols]
        lbp_image += (neighbor >= center).astype(np.uint8) << i
    
    return lbp_image
