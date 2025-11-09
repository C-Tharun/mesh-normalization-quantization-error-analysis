Mesh Normalization, Quantization, and Error Analysis – Final Report
Abstract

This project focuses on preparing 3D mesh data by normalizing, quantizing, and reconstructing it to analyze how much information is lost during these steps. Two normalization techniques — Min–Max and Unit Sphere — were tested to see how they affect reconstruction quality after quantization. The work also explores how meshes behave under transformations like rotation and translation, and how adaptive quantization can help improve precision in denser regions of the mesh.

Methodology
1. Normalization

Before working with 3D meshes, it’s important to bring all vertex coordinates to a consistent scale. In this project, I tried two common normalization methods.

Min–Max Normalization

Formula: v_norm = (v - v_min) / (v_max - v_min)

This maps all vertex coordinates to a [0, 1] range for each axis.

It preserves the overall proportions and is simple to implement.

The v_min and v_max values for each axis are stored for later reconstruction.

Unit Sphere Normalization

Steps:

Move the mesh so that its center (centroid) is at the origin.

Scale the mesh so that it fits inside a unit sphere.

This method is useful when meshes may be rotated or translated differently, since it focuses on the shape rather than absolute position.

The centroid and max_radius are saved to restore the original scale later.

2. Quantization

After normalization, the coordinates were quantized — which means converting continuous values into discrete bins.

Formula: q = floor(x_norm * (bins - 1))

Each axis was divided into 1024 bins, giving integer values from 0 to 1023.

For the Unit Sphere method, values were first mapped from [-1, 1] to [0, 1] before quantization.

3. Reconstruction

To check the effect of these transformations, I reversed the process to reconstruct the original mesh:

Dequantization: x_deq = q / (bins - 1)
Converts integers back into normalized values between [0, 1].

Denormalization:

For Min–Max: x_denorm = norm * (v_max - v_min) + v_min

For Unit Sphere: x_denorm = (norm * max_radius) + centroid

4. Error Evaluation

To measure how accurate the reconstruction was, I calculated two common error metrics:

MSE (Mean Squared Error): Average of squared differences between original and reconstructed coordinates.

MAE (Mean Absolute Error): Average of absolute differences.

Both metrics were calculated per axis (X, Y, Z) and as an overall average.

Results
Key Numeric Metrics

Tests were done on a simple cube mesh (2×2×2 units, 8 vertices, 12 faces) using 1024 quantization bins.

Min–Max Normalization

Normalized range: [0.0, 1.0]

Quantized range: [0, 1023]

MSE Overall: ~1e-6

MAE Overall: ~1e-3

Errors were almost identical across all three axes.

Unit Sphere Normalization

Normalized range: [-1.0, 1.0] (max radius = 1.0)

Quantized range: [0, 1023]

MSE Overall: ~4e-6

MAE Overall: ~1.6e-3

Slightly higher error due to scaling within a sphere, but still very small.

Observations

Reconstruction Quality:
Both normalization methods produced extremely small reconstruction errors, which means quantization with 1024 bins preserves mesh structure very well.

Error Distribution:
Errors were evenly spread across all axes for Min–Max normalization, while Unit Sphere showed very tiny variations because of its spherical scaling.

Quantization Effect:
Increasing the number of bins improves accuracy. In general, doubling the bins roughly halves the error (logarithmic relationship).

Bonus Research Results
Rotation & Translation Invariance

To test transformation robustness, I generated three random rotations and translations of the cube and checked if normalization produced similar results.

Unit Sphere Normalization:
It remained almost unchanged under rotation and translation. The mean difference was below 1e-6, confirming that it’s rotation/translation invariant.

Min–Max Normalization:
Not invariant — the normalized values depend on the bounding box, which changes with rotation.

Adaptive Quantization

I also experimented with adaptive quantization based on vertex density using k-nearest neighbor distance.

Dense regions received more bins (higher precision).

Sparse regions used fewer bins.

For the cube (which is uniform), the density was constant, so the improvement was negligible, but the approach could be useful for more complex meshes.

Uniform quantization (1024 bins): Baseline error ≈ 0

Adaptive quantization: Similar performance for uniform meshes, but potentially better for non-uniform geometries.

Conclusion
Which Normalization Method Worked Best?

Both Min–Max and Unit Sphere normalization gave nearly identical reconstruction results with errors close to zero.

Min–Max is simpler and works well when the mesh is already aligned and scaled consistently.

Unit Sphere is better when the mesh might be rotated, translated, or needs to be analyzed regardless of its position or scale.

Main Takeaways

Both methods preserve the original structure extremely well.
The tiny errors come mainly from quantization rounding, not the normalization method.

Min–Max normalization is ideal for consistent, pre-aligned datasets.

Unit Sphere normalization is useful for general-purpose or pose-invariant 3D applications.

Adaptive quantization is a promising idea for meshes with non-uniform vertex distributions, although it doesn’t make much difference for uniform shapes like cubes.

Overall Reflection

This project helped me understand how essential preprocessing is before feeding 3D data into AI or geometry-based models. Even though the math behind normalization and quantization seems simple, their impact on model accuracy and data consistency is huge.

Working through this also showed how errors propagate and how scaling and bin size can directly influence data quality — lessons that are useful for future work in areas like 3D reconstruction and graphics AI systems such as SeamGPT.

Visual Results

All visualization outputs are saved in the output/ folder:

error_minmax.png – Min–Max error metrics

error_unitsphere.png – Unit Sphere error metrics

error_comparison.png – Both methods side-by-side

adaptive_quantization_error.png – Uniform vs Adaptive comparison

rotation_invariance_comparison.png – Rotation invariance test results

Future Work

Test the pipeline on more complex meshes (organic or high-polygon models).

Compare multiple quantization levels (e.g., 64, 128, 256, 512, 1024).

Improve adaptive quantization for better performance on non-uniform meshes.

Experiment with other normalization techniques like PCA-based or z-score.

Measure compression ratios to study storage and transmission efficiency.

Report Prepared by: Tharun Subramanian C
Project: Mesh Normalization, Quantization, and Error Analysis
