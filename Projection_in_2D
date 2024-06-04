import open3d as o3d
import numpy as np
import pandas as pd
import openpyxl as px
from scipy.spatial import cKDTree

def apply_low_pass_filter_to_point_cloud(pcd_path):
    # 点群データを読み込み
    pcd = o3d.io.read_point_cloud(pcd_path)
    if pcd.is_empty():
        raise FileNotFoundError("点群ファイルがない")

    # numpy配列に変換
    points = np.asarray(pcd.points)  # 点の座標
    colors = np.asarray(pcd.colors)  # 色情報

    # 各軸のデータを取得
    x_data, y_data, z_data = points[:, 0], points[:, 1], points[:, 2]

    # XY平面に投影した2次元配列を作成し、Z情報を保持する
    resolution = 0.01  # 解像度の設定
    x_edges = np.arange(x_data.min(), x_data.max() + resolution, resolution)
    y_edges = np.arange(y_data.min(), y_data.max() + resolution, resolution)

    # 2Dヒストグラムを作成し、各ビンに絶対値が最大のZ値を保持する
    z_grid = np.full((len(x_edges) - 1, len(y_edges) - 1), np.nan)
    for i in range(len(x_data)):
        x_idx = np.searchsorted(x_edges, x_data[i]) - 1
        y_idx = np.searchsorted(y_edges, y_data[i]) - 1
        if np.isnan(z_grid[x_idx, y_idx]) or abs(z_data[i]) > abs(z_grid[x_idx, y_idx]):
            z_grid[x_idx, y_idx] = z_data[i]

    return z_grid, x_edges, y_edges

def apply_low_pass_filter(z_grid, filter_size):
    # フーリエ変換を実行
    f = np.fft.fft2(np.nan_to_num(z_grid))
    # ゼロ周波数成分を中央にシフト
    fshift = np.fft.fftshift(f)
    
    # マスクの作成
    rows, cols = z_grid.shape
    crow, ccol = int(rows / 2), int(cols / 2)

    mask = np.zeros((rows, cols), np.uint8)
    mask[crow-filter_size:crow+filter_size, ccol-filter_size:ccol+filter_size] = 1

    # フーリエ変換結果にマスクを適用
    fshift_masked = fshift * mask

    # 逆フーリエ変換で画像を復元
    f_ishift = np.fft.ifftshift(fshift_masked)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.real(np.fft.ifft2(f_ishift))

    return img_back

def remove_non_existed_points(points, original_points):
    # KDTreeを使用して近傍探索を行う
    tree = cKDTree(original_points)
    distances, indices = tree.query(points, distance_upper_bound=0.01)  # 距離の閾値を設定

    # 有効な点のみをフィルタリング
    valid_points = points[distances != np.inf]

    return valid_points

def regenerate_point_cloud(x_edges, y_edges, filtered_z_grid, original_points):
    # 新しい点群の座標を計算
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    x_coords, y_coords = np.meshgrid(x_centers, y_centers, indexing='ij')
    
    # フィルタ適用後のZ値を使用して点群を再生成
    valid_mask = filtered_z_grid != 0  # Z値が0でない部分のみを使用
    new_points = np.vstack((x_coords[valid_mask], y_coords[valid_mask], filtered_z_grid[valid_mask])).T

    # もともと存在しなかった点を削除
    new_points = remove_non_existed_points(new_points, original_points)


    # 新しい点群を作成
    new_pcd = o3d.geometry.PointCloud()
    new_pcd.points = o3d.utility.Vector3dVector(new_points)
    
    return new_pcd

# 使用例
pcd_path = "/home/riku-suzuki/python_tutorial/tottori_map_50.pcd"
filter_size = 1000000000000000  # フィルターサイズを適宜調整

# 元の点群を読み込み
pcd = o3d.io.read_point_cloud(pcd_path)
# フィルタ適用後の点群に座標軸を追加
coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0)
o3d.visualization.draw_geometries([pcd, coordinate_frame])

# フィルタ適用
z_grid, x_edges, y_edges = apply_low_pass_filter_to_point_cloud(pcd_path)
filtered_z_grid = apply_low_pass_filter(z_grid, filter_size)

# 点群を再生成
new_pcd = regenerate_point_cloud(x_edges, y_edges, filtered_z_grid, np.asarray(pcd.points))


# 座標軸と点群データを別々に描画
o3d.visualization.draw_geometries([new_pcd, coordinate_frame])


# 点群データのRGBと位置をエクセルファイルとして出力
#new_points = np.asarray(new_pcd.points)
#data = {
#    'x': new_points[:, 0],
#    'y': new_points[:, 1],
#    'z': new_points[:, 2]
#}


#df = pd.DataFrame(data)
#df.to_excel("filtered_point_cloud.xlsx", index=False)
