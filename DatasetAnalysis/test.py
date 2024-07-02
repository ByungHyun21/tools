import numpy as np
from scipy.spatial import ConvexHull

def normalize(vectors):
    if vectors.ndim == 1:
        norms = np.linalg.norm(vectors)
        return vectors / norms
    else:
        norms = np.linalg.norm(vectors, axis=1).reshape(-1, 1)
        return vectors / norms

def icosahedron_vertices():
    t = (1.0 + np.sqrt(5.0)) / 2.0
    vertices = np.array([
        [-1, t, 0],
        [1, t, 0],
        [-1, -t, 0],
        [1, -t, 0],
        [0, -1, t],
        [0, 1, t],
        [0, -1, -t],
        [0, 1, -t],
        [t, 0, -1],
        [t, 0, 1],
        [-t, 0, -1],
        [-t, 0, 1]
    ])
    return normalize(vertices)

def icosahedron_faces():
    faces = np.array([
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1]
    ])
    return faces

def midpoint(p1, p2):
    return normalize((p1 + p2) / 2.0)

def subdivide(vertices, faces):
    vertex_map = {}
    new_faces = []
    vertices_list = vertices.tolist()

    def get_vertex(p1, p2):
        key = tuple(sorted((tuple(p1), tuple(p2))))
        if key not in vertex_map:
            vertex_map[key] = len(vertices_list)
            vertices_list.append(midpoint(np.array(p1), np.array(p2)).tolist())
        return vertex_map[key]

    for face in faces:
        v1, v2, v3 = face
        a = get_vertex(vertices[v1], vertices[v2])
        b = get_vertex(vertices[v2], vertices[v3])
        c = get_vertex(vertices[v3], vertices[v1])
        new_faces.append([v1, a, c])
        new_faces.append([v2, b, a])
        new_faces.append([v3, c, b])
        new_faces.append([a, b, c])

    return np.array(vertices_list), np.array(new_faces)

def generate_geodesic_grid(subdivisions):
    vertices = icosahedron_vertices()
    faces = icosahedron_faces()

    for _ in range(subdivisions):
        vertices, faces = subdivide(vertices, faces)

    return vertices, faces

subdivisions = 2  # Subdivision level
vertices, faces = generate_geodesic_grid(subdivisions)

# Visualization
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot vertices
ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], color='r')

# Plot faces
mesh = Poly3DCollection(vertices[faces], edgecolor='k', alpha=0.3)
ax.add_collection3d(mesh)

plt.show()
