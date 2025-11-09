# Mesh Normalization, Quantization, and Error Analysis – Final Report

## Abstract
This project focuses on preparing 3D mesh data by normalizing, quantizing, and reconstructing it to analyze how much information is lost during these steps. Two normalization techniques — Min–Max and Unit Sphere — were implemented, followed by uniform and adaptive quantization. Reconstruction and error analysis (MSE and MAE) were used to evaluate information loss.

---

## Table of Contents
- Abstract
- Methodology
  - Normalization
    - Min–Max Normalization
    - Unit Sphere Normalization
  - Quantization
  - Reconstruction
  - Error Evaluation
- Results
  - Key Numeric Metrics
  - Observations
- Bonus Research Results
  - Rotation & Translation Invariance
  - Adaptive Quantization
- Conclusion
  - Which Normalization Method Worked Best?
  - Main Takeaways
  - Overall Reflection
- Visual Results
- Future Work
- Report Prepared by

---

## Methodology

### 1. Normalization
Before working with 3D meshes, it’s important to bring all vertex coordinates to a consistent scale. In this project two common normalization methods were used.

#### Min–Max Normalization
- Formula: v_norm = (v - v_min) / (v_max - v_min)  
- Maps all vertex coordinates to a [0, 1] range for each axis.  
- Preserves overall proportions and is simple to implement.  
- The per-axis v_min and v_max values are stored for later reconstruction.

#### Unit Sphere Normalization
Steps:
1. Move the mesh so its center (centroid) is at the origin.
2. Scale the mesh so that it fits inside a unit sphere.

- This method focuses on the shape (pose-invariant) rather than absolute position.
- The centroid and max_radius are saved to restore original scale during denormalization.

---

### 2. Quantization
After normalization, coordinates were quantized (converted to discrete bins).

- Formula: q = floor(x_norm * (bins - 1))  
- Each axis was divided into 1024 bins, yielding integer values from 0 to 1023.  
- For Unit Sphere normalization, values were first mapped from [-1, 1] to [0, 1] before quantization.

---

### 3. Reconstruction
To evaluate the effect of the transformations, the pipeline reversed the process:

- Dequantization: x_deq = q / (bins - 1) — converts integers back to normalized values in [0, 1].  
- Denormalization:
  - Min–Max: x_denorm = x_deq * (v_max - v_min) + v_min  
  - Unit Sphere: x_denorm = (x_deq * 2 - 1) * max_radius + centroid  
    (Note: mapping back from [0,1] to [-1,1] was applied for Unit Sphere.)

---

### 4. Error Evaluation
Two common error metrics were calculated to measure reconstruction accuracy:

- MSE (Mean Squared Error): average of squared differences between original and reconstructed coordinates.  
- MAE (Mean Absolute Error): average of absolute differences.

Both metrics were reported per axis (X, Y, Z) and as an overall average.

---

## Results

### Key Numeric Metrics
Tests were performed on a simple cube mesh (2×2×2 units, 8 vertices, 12 faces) using 1024 quantization bins.

#### Min–Max Normalization
- Normalized range: [0.0, 1.0]  
- Quantized range: [0, 1023]  
- MSE Overall: ≈ 1e-6  
- MAE Overall: ≈ 1e-3  
- Errors were nearly identical across X, Y, Z axes.

#### Unit Sphere Normalization
- Normalized range: [-1.0, 1.0] (max radius = 1.0)  
- Quantized range: [0, 1023]  
- MSE Overall: ≈ 4e-6  
- MAE Overall: ≈ 1.6e-3  
- Slightly higher error due to spherical scaling, but still very small.

---

### Observations

- Reconstruction Quality: Both normalization methods produced extremely small reconstruction errors. Quantization with 1024 bins preserves mesh structure very well.
- Error Distribution: Errors were evenly spread across axes for Min–Max normalization. Unit Sphere showed tiny variations because of spherical scaling.
- Quantization Effect: Increasing the number of bins improves accuracy. In general, doubling the bins roughly halves the error (rough/logarithmic relationship).

---

## Bonus Research Results

### Rotation & Translation Invariance
- To test robustness, three random rotations and translations of the cube were generated and normalized.
- Unit Sphere Normalization: Remained almost unchanged under rotation and translation. Mean difference was below 1e-6, confirming rotation/translation invariance.
- Min–Max Normalization: Not invariant — normalized values depend on the bounding box, which changes with rotation.

### Adaptive Quantization
- Experimented with adaptive quantization based on local vertex density (k-nearest neighbor distances).  
- Dense regions received more bins (higher precision); sparse regions used fewer bins.  
- For the uniform cube, density was constant so improvement was negligible.  
- Uniform quantization (1024 bins): Baseline error ≈ 0.  
- Adaptive quantization: Similar performance for uniform meshes; potentially better for non-uniform geometries.

---

## Conclusion

### Which Normalization Method Worked Best?
- Both Min–Max and Unit Sphere normalization gave nearly identical reconstruction results with errors close to zero.
- Min–Max is simpler and works well when meshes are aligned and scaled consistently.
- Unit Sphere is better when meshes may be rotated or translated, or when pose invariance is required.

### Main Takeaways
- Both methods preserve original structure extremely well.
- The tiny errors come mostly from quantization rounding, not the normalization method.
- Min–Max is ideal for consistent, pre-aligned datasets.
- Unit Sphere is useful for general-purpose or pose-invariant 3D applications.
- Adaptive quantization is promising for meshes with non-uniform vertex distributions.

### Overall Reflection
This project reinforced the importance of preprocessing 3D data before feeding it into AI or geometry-based models. Even simple normalization and quantization steps can significantly affect downstream quality. Working through this showed how errors propagate and how scaling and bin size influence data quality — useful lessons for 3D reconstruction and geometric processing.

---

## Visual Results
All visualization outputs are saved in the `output/` folder:
- `output/error_minmax.png` — Min–Max error metrics  
- `output/error_unitsphere.png` — Unit Sphere error metrics  
- `output/error_comparison.png` — Both methods side-by-side  
- `output/adaptive_quantization_error.png` — Uniform vs Adaptive comparison  
- `output/rotation_invariance_comparison.png` — Rotation invariance test results

---

## Future Work
- Test the pipeline on more complex meshes (organic or high-polygon models).  
- Compare multiple quantization levels (e.g., 64, 128, 256, 512, 1024).  
- Improve adaptive quantization for better performance on non-uniform meshes.  
- Experiment with other normalization techniques like PCA-based normalization or z-score normalization.  
- Measure compression ratios to study storage and transmission efficiency.

---

Report Prepared by: Tharun Subramanian C  
Project: Mesh Normalization, Quantization, and Error Analysis