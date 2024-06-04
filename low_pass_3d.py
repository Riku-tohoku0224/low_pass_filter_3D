import open3d as o3d
import numpy as np

def apply_low_pass_filter_to_point_cloud(pcd_path, filter_size):
    # 点群データを読み込み
    pcd = o3d.io.read_point_cloud(pcd_path)
    if pcd.is_empty():
        raise FileNotFoundError("点群ファイルがない")

    # numpy配列に変換
    points = np.asarray(pcd.points)  # 点の座標
    colors = np.asarray(pcd.colors)  # 色情報

    # 各軸のデータを取得
    x_data, y_data, z_data = points[:, 0], points[:, 1], points[:, 2]

    

    # 各軸ごとにフーリエ変換を適用する関数
    def low_pass_filter(data, filter_size):
        # 1次元フーリエ変換を実行
        f = np.fft.fft(data)
        fshift = np.fft.fftshift(f)

        # マスクの作成
        mask = np.zeros(data.shape, dtype=bool)
        center = data.shape[0] // 2
        mask[center - filter_size:center + filter_size] = True

        # フィルタを適用
        fshift_filtered = fshift * mask

        # フィルタリング前後の非ゼロ要素のバイト数を計算
        original_nonzero_elements = np.count_nonzero(fshift)
        filtered_nonzero_elements = np.count_nonzero(fshift_filtered)
        element_size = fshift.itemsize

        original_nonzero_bytes = original_nonzero_elements * element_size
        filtered_nonzero_bytes = filtered_nonzero_elements * element_size

        print(f"Original non-zero frequency data size: {original_nonzero_bytes} bytes")
        print(f"Filtered non-zero frequency data size: {filtered_nonzero_bytes} bytes")

        # 逆フーリエ変換
        f_ishift = np.fft.ifftshift(fshift_filtered)
        filtered_data = np.fft.ifft(f_ishift).real

        return filtered_data

    # 各軸にフィルタを適用
    filtered_x = low_pass_filter(x_data, filter_size)
    filtered_y = low_pass_filter(y_data, filter_size)
    filtered_z = low_pass_filter(z_data, filter_size)

    # フィルタ適用後の点群を作成
    filtered_points = np.vstack((filtered_x, filtered_y, filtered_z)).T

    # 新しい点群を作成
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
    filtered_pcd.colors = o3d.utility.Vector3dVector(colors[:filtered_points.shape[0], :])  # 色情報を適用

    # 元の点群を表示
    o3d.visualization.draw_geometries([pcd], window_name="Original Point Cloud")

    # フィルタ適用後の点群を表示
    o3d.visualization.draw_geometries([filtered_pcd], window_name="Filtered Point Cloud")

# フィルターの閾値を設定
filter_size = 60000  # フィルターのサイズを適宜調整
apply_low_pass_filter_to_point_cloud("/home/riku-suzuki/python_tutorial/tottori_map_50.pcd", filter_size)
