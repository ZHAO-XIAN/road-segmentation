import numpy as np
from collections import deque

def bfs_remove_noise_optimized(image, noise_threshold):
    """
    使用 BFS 清除影像中的噪點 (優化版本)

    參數:
    image (numpy.ndarray): 單通道二值影像，噪點為白色（255），背景為黑色（0）
    noise_threshold (int): 定義噪點的最大連通區域大小

    返回:
    numpy.ndarray: 已移除噪點的影像
    """
    rows, cols = image.shape
    visited = np.zeros((rows, cols), dtype=bool)  # 記錄訪問過的像素
    output_image = image.copy()  # 建立輸出影像副本

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 四個方向

    for i in range(rows):
        for j in range(cols):
            if image[i, j] == 255 and not visited[i, j]:  # 若是未訪問的白色像素
                queue = deque([(i, j)])
                connected_pixels = [(i, j)]
                visited[i, j] = True

                # 執行 BFS 搜尋區域
                while queue:
                    cx, cy = queue.popleft()
                    
                    for dx, dy in directions:
                        nx, ny = cx + dx, cy + dy
                        if 0 <= nx < rows and 0 <= ny < cols and not visited[nx, ny] and image[nx, ny] == 255:
                            queue.append((nx, ny))
                            connected_pixels.append((nx, ny))
                            visited[nx, ny] = True

                # 如果區域的像素數量小於閾值，將該區域視為噪點
                if len(connected_pixels) < noise_threshold:
                    # 將噪點區域設為背景色（黑色）
                    for x, y in connected_pixels:
                        output_image[x, y] = 0

    return output_image
