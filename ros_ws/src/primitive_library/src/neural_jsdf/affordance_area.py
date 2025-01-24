"""
@Author: Yiting CHEN
@Email: chenyiting@whu.edu.cn
"""
import open3d as o3d
import numpy as np

pcd = o3d.io.read_point_cloud("objects/banana/visual.ply")
# pcd = pcd.uniform_down_sample(10)
print(len(pcd.points))
pcd_mesh = o3d.io.read_triangle_mesh("objects/banana/textured.obj")
pcd.estimate_normals()


assert (pcd.has_normals())

# using all defaults
oboxes = pcd.detect_planar_patches(
    normal_variance_threshold_deg=70,
    coplanarity_deg=75,
    outlier_ratio=0.75,
    min_plane_edge_length=0.1,
    min_num_points=20,
    search_param=o3d.geometry.KDTreeSearchParamKNN(knn=50))

print("Detected {} patches".format(len(oboxes)))

geometries = []
for i, obox in enumerate(oboxes):
    obox_pcd = o3d.geometry.PointCloud()
    obox_pcd.points = o3d.utility.Vector3dVector([obox.center])

    dist = np.asarray(pcd.compute_point_cloud_distance(obox_pcd))
    near_points_index = np.argsort(dist)[::1][:100]

    points = pcd.select_by_index(near_points_index)
    points.paint_uniform_color(obox.color)
    o3d.io.write_point_cloud('objects/banana/area_{}.ply'.format(i), points)
    geometries.append(points)
    # geometries.append(obox)
geometries.append(pcd_mesh)

o3d.visualization.draw_geometries(geometries)