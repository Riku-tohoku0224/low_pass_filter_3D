import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

def apply_low_pass_filter_to_point_cloud(pcd_path, resolution=0.01):
    pcd = o3d.io.read_point_cloud(pcd_path)
    if pcd.is_empty():
        raise FileNotFoundError("点群ファイルがない")

    points = np.asarray(pcd.points)
    x_data, y_data, z_data = points[:, 0], points[:, 1], points[:, 2]

    x_edges = np.arange(x_data.min(), x_data.max() + resolution, resolution)
    y_edges = np.arange(y_data.min(), y_data.max() + resolution, resolution)

    z_grid = np.full((len(x_edges) - 1, len(y_edges) - 1, 2), np.nan)
    for i in range(len(x_data)):
        x_idx = np.searchsorted(x_edges, x_data[i]) - 1
        y_idx = np.searchsorted(y_edges, y_data[i]) - 1
        if np.isnan(z_grid[x_idx, y_idx, 0]) or abs(z_data[i]) > abs(z_grid[x_idx, y_idx, 0]):
            z_grid[x_idx, y_idx, 0] = z_data[i]
            z_grid[x_idx, y_idx, 1] = 1

    z_grid[np.isnan(z_grid[:, :, 0]), 1] = 0

    return z_grid, x_edges, y_edges

def apply_low_pass_filter(z_grid, filter_size):
    rows, cols, _ = z_grid.shape
    f = np.fft.fft2(np.nan_to_num(z_grid[:,:,0]))
    fshift = np.fft.fftshift(f)

    crow, ccol = int(rows / 2), int(cols / 2)
    mask = np.zeros((rows, cols), np.uint8)
    for i in range(rows):
        for j in range(cols):
            if (i - crow)**2 + (j - ccol)**2 <= filter_size**2:
                mask[i, j] = 1

    fshift_masked = fshift * mask
    f_ishift = np.fft.ifftshift(fshift_masked)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.real(img_back)

    filtered_z_grid = np.zeros_like(z_grid)
    filtered_z_grid[:,:,0] = img_back
    filtered_z_grid[:,:,1] = z_grid[:,:,1]

    non_zero_before = np.count_nonzero(fshift)
    non_zero_after = np.count_nonzero(fshift_masked)

    print(f"Data size in frequency domain before filtering: {non_zero_before}")
    print(f"Data size in frequency domain after filtering : {non_zero_after}")

    # 描画部分
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.title("Original Fourier Transform")
    plt.imshow(np.log(np.abs(f) + 1), cmap='gray')
    plt.colorbar()

    plt.subplot(2, 2, 2)
    plt.title("Shifted Fourier Transform")
    plt.imshow(np.log(np.abs(fshift) + 1), cmap='gray')
    plt.colorbar()

    plt.subplot(2, 2, 3)
    plt.title("Masked Fourier Transform")
    plt.imshow(np.log(np.abs(fshift_masked) + 1), cmap='gray')
    plt.colorbar()

    plt.subplot(2, 2, 4)
    plt.title("Filtered Inverse Fourier Transform")
    plt.imshow(img_back, cmap='gray')
    plt.colorbar()

    plt.tight_layout()
    plt.show()

    return filtered_z_grid

def regenerate_point_cloud(x_edges, y_edges, filtered_z_grid):
    mask = filtered_z_grid[:,:,1].ravel() == 1
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    x_coords, y_coords = np.meshgrid(x_centers, y_centers, indexing='ij')

    new_points = np.vstack((x_coords.ravel()[mask], y_coords.ravel()[mask], filtered_z_grid[:,:,0].ravel()[mask])).T

    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(new_points)

    return filtered_pcd

pcd_path = r"C:\Users\riku0\Downloads\tottori_map_50.pcd"
filter_size = 500 #[LP/mm]や[cycle/mm]

# 点群を読み込む
pcd = o3d.io.read_point_cloud(pcd_path)

# カラー属性を削除する
if pcd.has_colors():
    pcd.colors = o3d.utility.Vector3dVector([])

# 座標フレームを作成する
coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)

# Visualizerを使って点群と座標フレームを表示する
vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(pcd)
vis.add_geometry(coordinate_frame)
vis.run()
vis.destroy_window()

z_grid, x_edges, y_edges = apply_low_pass_filter_to_point_cloud(pcd_path)
filtered_z_grid = apply_low_pass_filter(z_grid, filter_size)

new_pcd = regenerate_point_cloud(x_edges, y_edges, filtered_z_grid)
o3d.visualization.draw_geometries([new_pcd, coordinate_frame])
