import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
from collections import defaultdict


def stitch_segments(segments):
    """
    Stitches a list of segments belonging to a single polygon into a closed loop.
    This is a simplified version for when all segments form one cycle.
    """
    graph = defaultdict(list)
    point_to_tuple = lambda p: tuple(np.round(p, 5))

    for seg in segments:
        p1, p2 = point_to_tuple(seg[0]), point_to_tuple(seg[1])
        graph[p1].append(p2)
        graph[p2].append(p1)

    start_node = next(iter(graph))
    path = [start_node]
    
    # Check if there's a path to follow
    if not graph[start_node]:
        return np.array([]) # Isolated point, no polygon
        
    prev_node = start_node
    current_node = graph[start_node][0]

    while current_node != start_node:
        path.append(current_node)
        neighbors = graph[current_node]
        
        # Find the next node that isn't the one we just came from
        if len(neighbors) == 1:
            # Dead end in an open polygon, break
            break
        
        next_node = neighbors[0] if neighbors[1] == prev_node else neighbors[1]
        prev_node = current_node
        current_node = next_node
    
    return np.array(path)



def colored_slice(vor_3d, bound=None, plane_axis='z', plane_value=0.0, cell_colors=None, path='', id=0):
    """
    Computes and plots the 2D cross-section of a 3D Voronoi diagram,
    coloring sliced polygons based on their parent 3D cell.
    """
    slicing_axis = {'x': 0, 'y': 1, 'z': 2}[plane_axis]

    # print(vor_3d)

    # 3. Group intersection segments by the cell they belong to
    cell_to_segments = defaultdict(list)

    for i, ridge_vertices in enumerate(vor_3d.ridge_vertices):
        if -1 in ridge_vertices:
            continue
        
        # Find the two cells this ridge separates
        p1_idx, p2_idx = vor_3d.ridge_points[i]

        face_vertices = vor_3d.vertices[ridge_vertices]
        distances = face_vertices[:, slicing_axis] - plane_value
        
        if distances.max() > 0 and distances.min() < 0:
            intersection_points_3d = []
            for j in range(len(face_vertices)):
                p1 = face_vertices[j]
                p2 = face_vertices[(j + 1) % len(face_vertices)]
                d1, d2 = distances[j], distances[(j + 1) % len(distances)]

                if d1 * d2 < 0:
                    t = d1 / (d1 - d2)
                    intersection_points_3d.append(p1 + t * (p2 - p1))
            
            if len(intersection_points_3d) == 2:
                p_start_2d = np.delete(intersection_points_3d[0], slicing_axis)
                p_end_2d = np.delete(intersection_points_3d[1], slicing_axis)
                segment = np.array([p_start_2d, p_end_2d])
                
                # Add this segment to both cells it borders
                cell_to_segments[p1_idx].append(segment)
                cell_to_segments[p2_idx].append(segment)

    # 4. Plot the results
    plt.figure(figsize=(20, 20))
    ax = plt.gca()

    # For each cell that was sliced...
    for cell_idx, segments in cell_to_segments.items():
        # Stitch its border segments into a polygon
        polygon = stitch_segments(segments)
        
        if polygon.size > 0:
            # Fill the polygon with the pre-assigned color of its parent 3D cell
            ax.fill(polygon[:, 0], polygon[:, 1], color=cell_colors[cell_idx], alpha=0.5)
            # Optionally, draw a black border for definition
            ax.plot(np.append(polygon[:, 0], polygon[0, 0]), np.append(polygon[:, 1], polygon[0, 1]), 'k-', lw=0.1)

    ax.set_title(f"Colored Cross-Section of 3D Voronoi Diagram at {plane_axis}={plane_value}")
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.show()
    plt.axis(bound)
    plt.show()
    plt.savefig('./'+path+'/voronoi_slice_'+str(id)+'.png')
    plt.close()


if __name__ == "__main__":
    num_points = 300
    # Assign a unique, random light color to each 3D cell (point)
    np.random.seed(0) # for reproducibility
    cell_colors = [np.random.rand(3) * 0.7 + 0.3 for _ in range(num_points)]
    np.random.seed(1)
    points_3d = np.random.rand(num_points, 3) * 10

    # 2. Compute the true slice of the 3D Voronoi diagram at z=5
    plane_slice_value = 4.05
    slice_line_segments = colored_slice(
        Voronoi(points_3d),
        plane_axis='z',
        plane_value=plane_slice_value,
        cell_colors=cell_colors
    )

# 3. Plot the results
# plt.figure(figsize=(10, 10))
# ax = plt.gca()

# # Plot the collected line segments
# for segment in slice_line_segments:
#     ax.plot(segment[:, 0], segment[:, 1], 'b-', lw=2)

# # For context, plot the points projected onto the plane
# # Color them based on whether they are above or below the slice
# projected_points = np.delete(points_3d, 2, axis=1)
# colors = ['red' if p[2] > plane_slice_value else 'black' for p in points_3d]
# ax.scatter(projected_points[:, 0], projected_points[:, 1], c=colors, s=50, zorder=10)

# ax.set_title(f"True Cross-Section of 3D Voronoi Diagram at z={plane_slice_value}")
# ax.set_aspect('equal', adjustable='box')
# ax.grid(True, linestyle='--', alpha=0.6)
# xmin, xmax = np.min(points_3d[:, 0]), np.max(points_3d[:, 0])
# ymin, ymax = np.min(points_3d[:, 1]), np.max(points_3d[:, 1])
# plt.axis([xmin, xmax, ymin, ymax])
# plt.show()
# plt.savefig('voronoi_slice.png')