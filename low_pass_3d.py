import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

def apply_low_pass_filter_to_point_cloud(pcd, base_height, resolution, filter_size):
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    x_data, y_data, z_data = points[:, 0], points[:, 1], points[:, 2]

    z_data = z_data

    x_edges = np.arange(x_data.min(), x_data.max() + resolution, resolution)
    y_edges = np.arange(y_data.min(), y_data.max() + resolution, resolution)

    z_grid = np.full((len(x_edges) - 1, len(y_edges) - 1, 4), np.nan, dtype=np.float32)
    valid_grid = np.zeros((len(x_edges) - 1, len(y_edges) - 1), dtype=bool)
    
    for i in range(len(x_data)):
        x_idx = np.searchsorted(x_edges, x_data[i]) - 1
        y_idx = np.searchsorted(y_edges, y_data[i]) - 1
        if np.isnan(z_grid[x_idx, y_idx, 0]) or abs(z_data[i]) > abs(z_grid[x_idx, y_idx, 0]):
            z_grid[x_idx, y_idx, 0] = z_data[i]
            z_grid[x_idx, y_idx, 1] = colors[i, 0]
            z_grid[x_idx, y_idx, 2] = colors[i, 1]
            z_grid[x_idx, y_idx, 3] = colors[i, 2]
            valid_grid[x_idx, y_idx] = True

    z_grid[np.isnan(z_grid[:, :, 0])] = base_height
    valid_grid[np.isnan(z_grid[:, :, 0])] = False

    rows, cols, _ = z_grid.shape

    valid_grid_memory_usage = valid_grid.size  # bool は 1 バイト
    f = np.fft.fft2(np.nan_to_num(z_grid[:,:,0]))
    fshift = np.fft.fftshift(f)
   
    crow, ccol = int(rows / 2), int(cols / 2)
    mask = np.zeros((rows, cols), np.uint8)
    for i in range(rows):
        for j in range(cols):
            if (i - crow)**2 + (j - ccol)**2 <= filter_size**2:
                mask[i, j] = 1

    fshift_masked = fshift * mask

    non_zero_before = np.count_nonzero(fshift)
    non_zero_after = np.count_nonzero(fshift_masked)
    fshift_total_memory_usage = non_zero_before * 8
    fshift_masked_total_memory_usage = non_zero_after * 8

    print(f"Total memory usage of valid_grid: {valid_grid_memory_usage} bytes")
    print(f"Total memory usage of fshift: {fshift_total_memory_usage} bytes")
    print(f"Total memory usage of fshift_masked: {fshift_masked_total_memory_usage} bytes")
    print(f"Data size before filtering: {fshift_total_memory_usage + valid_grid_memory_usage} bytes")
    print(f"Data size after filtering : {fshift_masked_total_memory_usage + valid_grid_memory_usage} bytes")

    # 描画部分
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.title("Original Fourier Transform")
    plt.imshow(np.abs(z_grid[:,:,0]), cmap='gray')
    plt.colorbar()

    plt.subplot(2, 2, 2)
    plt.title("Shifted Fourier Transform")
    plt.imshow(np.log(np.abs(fshift) + 1), cmap='gray')
    plt.colorbar()

    plt.subplot(2, 2, 3)
    plt.title("Masked Fourier Transform")
    plt.imshow(np.log(np.abs(fshift_masked) + 1), cmap='gray')
    plt.colorbar()

    plt.tight_layout()
    plt.show()



    return fshift_masked, x_edges, y_edges, z_grid, valid_grid

def regenerate_point_cloud(fshift_masked, x_edges, y_edges, z_grid, valid_grid, base_height):
    f_ishift = np.fft.ifftshift(fshift_masked)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.real(img_back) 

    filtered_z_grid = np.zeros_like(z_grid)
    filtered_z_grid[:,:,0] = img_back
    filtered_z_grid[:,:,1:] = z_grid[:,:,1:]

    mask = valid_grid.ravel()
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    x_coords, y_coords = np.meshgrid(x_centers, y_centers, indexing='ij')

    new_points = np.vstack((x_coords.ravel()[mask], y_coords.ravel()[mask], filtered_z_grid[:,:,0].ravel()[mask])).T
    new_colors = np.vstack((filtered_z_grid[:,:,1].ravel()[mask], filtered_z_grid[:,:,2].ravel()[mask], filtered_z_grid[:,:,3].ravel()[mask])).T

    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(new_points)
    filtered_pcd.colors = o3d.utility.Vector3dVector(new_colors)
     # 描画部分
    plt.figure(figsize=(12, 8))



    plt.subplot(2, 2, 4)
    plt.title("Filtered Inverse Fourier Transform")
    plt.imshow(img_back, cmap='gray')
    plt.colorbar()

    plt.tight_layout()
    plt.show()


    return filtered_pcd


pcd_path = "/home/riku-suzuki/madmax/test/E-2/threshold_100/map.pcd"

# 点群を読み込む
pcd = o3d.io.read_point_cloud(pcd_path)

# カラー情報が正しく読み込まれているか確認
if not pcd.has_colors():
    print("No color information found in PCD file.")
else:
    # カラー情報を0-1の範囲にスケール
    colors = np.asarray(pcd.colors)
    if colors.max() > 1.0:
        colors = colors / 255.0

    # カラー情報を表示（BGR -> RGBに変換）
    colors = colors[:, [2, 1, 0]]  # BGRからRGBに順番を変更

    # ポイントクラウドのカラー情報を更新
    pcd.colors = o3d.utility.Vector3dVector(colors)

# 座標フレームを作成する
coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)

# 初回の可視化
def visualize_point_cloud(pcd, coordinate_frame):
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.add_geometry(coordinate_frame)
    
    # 点のサイズを変更
    render_option = vis.get_render_option()
    render_option.point_size = 0.05  # 点のサイズを小さくする

    vis.run()
    vis.destroy_window()

#visualize_point_cloud(pcd, coordinate_frame)

# 重心の計算
centroid = np.asarray(pcd.points).mean(axis=0)
height_every_frame = centroid[2]

resolution = 0.1
filter_size = 100
fshift_masked, x_edges, y_edges, z_grid, valid_grid = apply_low_pass_filter_to_point_cloud(pcd, height_every_frame, resolution, filter_size)
regenerated_pcd = regenerate_point_cloud(fshift_masked, x_edges, y_edges, z_grid, valid_grid, height_every_frame)

# 二回目の可視化
visualize_point_cloud(regenerated_pcd, coordinate_frame)

#output_path = f"/home/riku-suzuki/madmax/test/E-2/static/regenerated_map_{filter_size}.pcd"

# 再生成された点群を保存する
#o3d.io.write_point_cloud(output_path, regenerated_pcd)
