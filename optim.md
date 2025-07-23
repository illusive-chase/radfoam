# 1. Densification

Densification is the process of adding new points (vertices) to the tetrahedral mesh to improve detail in areas with high rendering error.


  1. Triggering Densification: The process is initiated within the main train_loop in train.py. It runs periodically between the densify_from and densify_until training iterations.


  2. Error Calculation: Before densification, an error map is computed for each point using model.collect_error_map. This function leverages the backward pass of the renderer (trace_backward in src/tracing/pipeline.cu) to calculate a point_error, which represents the volumetric rendering error associated with each vertex.

  3. Candidate Selection: Points with the highest rendering error are selected as candidates for densification.


  4. Splitting Mechanism: As described in the paper, for each high-error tetrahedron, RadFoam splits its longest edge. The prune_and_densify method (in radfoam_model/scene.py, inferred from train.py) identifies these edges and adds their midpoints as new vertices to the point cloud.


  5. Triangulation Update: After adding the new points, the entire 3D Delaunay triangulation is rebuilt by calling model.update_triangulation(incremental=False). This calls into the C++/CUDA backend, specifically the Triangulation::rebuild method (src/delaunay/delaunay.cu), to create a new valid tetrahedral mesh incorporating the new points.

# 2. Culling

Culling is the process of removing vertices and tetrahedra that are unnecessary or contribute little to the final render. This happens in two ways: contribution-based pruning and geometric culling.


  1. Contribution-Based Pruning:
      * Trigger: This occurs alongside densification in the prune_and_densify method.
      * Metric: The collect_error_map function in train.py also computes a point_contribution metric for each vertex. This is done during the forward pass (trace_forward in src/tracing/pipeline.cu), which tracks how much each vertex contributes to the final pixel colors.
      * Mechanism: Vertices whose contribution is below a certain threshold are removed from the point set. The paper also mentions that points with an exceptionally large Voronoi cell radius (indicating they are outliers) are culled. The farthest_neighbor kernel (src/delaunay/triangulation_ops.cu) is used to compute this radius.


  2. Geometric Culling (Delaunay Violations):
      * Trigger: This is an integral part of maintaining the mesh's geometric integrity and happens during the triangulation rebuild process.
      * Metric: The Delaunay condition itself is the metric. A tetrahedron is valid only if its circumsphere contains no other points from the set.
      * Mechanism: The delete_delaunay_violations function (src/delaunay/delete_violations.cu) is called during triangulation. It uses the check_delaunay_warp kernel (src/delaunay/exact_tree_ops.cuh) to test each tetrahedron. Any tetrahedron that fails the test (i.e., is non-Delaunay) is culled from the mesh. This ensures the triangulation remains a valid Delaunay triangulation after point positions are updated during optimization.