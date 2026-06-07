from app.visualization.tsne_visualizer import TSNEVisualizer

visualizer = TSNEVisualizer()

coords_2d, _ = visualizer.generate_2d()

print(coords_2d.shape)

coords_3d, _ = visualizer.generate_3d()

print(coords_3d.shape)