import trimesh


object_file = 'champagne_glass.obj'


def main() -> None:

    print("Loading mesh")
    full_mesh = trimesh.load(object_file, force='mesh')

    print("Showing original mesh")
    full_mesh.show()


    convex_mesh = full_mesh.convex_hull
    convex_mesh.show()
    file_name = f"champagne_glass_collider.obj"
    with open(file_name, 'w') as f:
        convex_mesh.export(f, file_type='obj')


    # print("Performing convex decomposition...")
    # nearly_convex_meshes = full_mesh.convex_decomposition()
    # convex_meshes = []
    # convex_scene = trimesh.scene.scene.Scene()
    # for msh in nearly_convex_meshes:
    #     convex_meshes.append(msh.convex_hull)
    #     convex_scene.add_geometry(msh.convex_hull)

    # print("Displaying result")
    # convex_scene.show()

    # for i, mesh in enumerate(convex_meshes):
    #     file_name = f"champagne_glass_collision_{i}.obj"
    #     with open(file_name, 'w') as f:
    #         mesh.export(f, file_type='obj')

if __name__ == "__main__":
    main()
