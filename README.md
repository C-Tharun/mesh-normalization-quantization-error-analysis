# Mesh Normalization, Quantization, and Error Analysis

## Project Overview

This project implements a comprehensive pipeline for 3D mesh normalization, quantization, and error analysis. It demonstrates two normalization methods (Min-Max and Unit Sphere), quantization techniques, and evaluates reconstruction errors. The project also includes bonus research on rotation/translation invariance and adaptive quantization.

## Objective

The main objective is to:
1. Load and inspect 3D mesh files (.obj format)
2. Normalize mesh vertices using different methods
3. Quantize normalized vertices to discrete integer values
4. Reconstruct meshes through dequantization and denormalization
5. Compute and analyze reconstruction errors
6. Visualize results and compare different methods

## Dependencies

The project requires the following Python packages:

- `trimesh` - For loading and manipulating 3D meshes
- `numpy` - For numerical computations
- `matplotlib` - For visualization and plotting
- `scipy` - For spatial data structures (KD-tree)

## Installation

1. Clone or download this repository
2. Install dependencies using pip:

```bash
pip install -r requirements.txt
```

Or install packages individually:

```bash
pip install trimesh numpy matplotlib scipy
```

## Project Structure

```
mesh-normalization-quantization-error-analysis/
│
├── main.py                 # Main script with all functions and tests
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── report.md              # Detailed report (for PDF export)
│
├── meshes/                # Input mesh files directory
│   └── sample_cube.obj    # Sample cube mesh (auto-generated)
│
└── output/                # Output directory (auto-created)
    ├── normalized_minmax.ply
    ├── quantized_minmax.ply
    ├── normalized_unitsphere.ply
    ├── quantized_unitsphere.ply
    ├── reconstructed_minmax.ply
    ├── reconstructed_unitsphere.ply
    ├── error_minmax.png
    ├── error_unitsphere.png
    ├── error_comparison.png
    ├── adaptive_quantization_error.png
    └── rotation_invariance_comparison.png
```

## Execution Guide

### Step 1: Prepare Input Mesh

The script automatically generates a sample cube mesh if it doesn't exist. To use your own mesh:

1. Place your `.obj` file in the `meshes/` directory
2. Update `INPUT_MESH_PATH` in `main.py` (line 22)

### Step 2: Run the Script

Execute the main script:

```bash
python main.py
```

This will run all tests automatically:
- Task 1: Load and inspect mesh
- Task 2: Normalize and quantize mesh
- Task 3: Dequantize, denormalize, and compute error
- Task 5: Bonus research (rotation invariance & adaptive quantization)

### Step 3: View Results

All output files are saved in the `output/` directory:
- `.ply` files: Mesh files (can be viewed in mesh viewers like MeshLab, Blender)
- `.png` files: Error visualization plots

## Tasks Summary

### Task 1: Load and Inspect Mesh

- Loads `.obj` mesh files using trimesh
- Extracts vertex coordinates as NumPy arrays
- Prints statistics:
  - Number of vertices and faces
  - Min, max, mean, and standard deviation per axis (X, Y, Z)
  - Bounding box information

**Sample Results:**
- Cube mesh: 8 vertices, 12 faces
- Vertex range: Typically [-1, 1] for a 2x2x2 cube

### Task 2: Normalize and Quantize

Implements two normalization methods:

1. **Min-Max Normalization**
   - Formula: `v_norm = (v - v_min) / (v_max - v_min)`
   - Normalizes to [0, 1] range
   - Quantizes using 1024 bins

2. **Unit Sphere Normalization**
   - Translates to origin (centroid)
   - Scales to unit sphere (max radius = 1)
   - Quantizes using 1024 bins

**Sample Results:**
- Min-Max normalized range: [0.0, 1.0]
- Unit Sphere normalized range: [-1.0, 1.0] (max radius ≤ 1.0)
- Quantized range: [0, 1023] (integers)

### Task 3: Dequantize, Denormalize, and Compute Error

1. **Dequantization**
   - Formula: `x_deq = q / (bins - 1)`
   - Converts integer bins back to [0, 1] range

2. **Denormalization**
   - Min-Max: `x_denorm = norm * (v_max - v_min) + v_min`
   - Unit Sphere: Scale by max_radius, then translate by centroid

3. **Error Computation**
   - Mean Squared Error (MSE) per axis and overall
   - Mean Absolute Error (MAE) per axis and overall

**Sample Results:**
- For cube with 1024 bins:
  - MSE Overall: ~1e-6 to 1e-5
  - MAE Overall: ~1e-3 to 1e-2
- Errors are typically very small due to high quantization precision

### Task 5: Bonus Research

1. **Rotation & Translation Invariance**
   - Tests if Unit Sphere normalization produces identical results after rotation/translation
   - Generates 3 random transformations
   - Compares normalized outputs (should be identical)

2. **Adaptive Quantization**
   - Computes local vertex density using k-nearest neighbors
   - Uses variable bin sizes: more bins in dense regions, fewer in sparse regions
   - Compares reconstruction errors with uniform quantization

**Sample Results:**
- Unit Sphere normalization: ✓ Rotation/translation invariant
- Adaptive quantization: May show different error patterns depending on mesh geometry

## Output Files

### Mesh Files (.ply)
- `normalized_minmax.ply` - Min-Max normalized mesh
- `quantized_minmax.ply` - Quantized mesh (Min-Max)
- `normalized_unitsphere.ply` - Unit Sphere normalized mesh
- `quantized_unitsphere.ply` - Quantized mesh (Unit Sphere)
- `reconstructed_minmax.ply` - Reconstructed mesh (Min-Max)
- `reconstructed_unitsphere.ply` - Reconstructed mesh (Unit Sphere)

### Visualization Files (.png)
- `error_minmax.png` - Error metrics for Min-Max normalization
- `error_unitsphere.png` - Error metrics for Unit Sphere normalization
- `error_comparison.png` - Side-by-side comparison of both methods
- `adaptive_quantization_error.png` - Uniform vs Adaptive quantization comparison
- `rotation_invariance_comparison.png` - Rotation invariance test results

## Example Commands

```bash
# Run all tests
python main.py

# View output files
ls output/

# View a specific mesh (requires mesh viewer)
# Open output/reconstructed_minmax.ply in MeshLab or Blender
```

## Key Functions

### Core Functions
- `load_and_inspect_mesh(file_path)` - Load and analyze mesh
- `normalize_minmax(vertices)` - Min-Max normalization
- `normalize_unitsphere(vertices)` - Unit Sphere normalization
- `quantize(vertices, bins=1024)` - Quantize vertices
- `dequantize(quantized_vertices, bins=1024)` - Dequantize vertices
- `denormalize_minmax(norm_vertices, v_min, v_max)` - Denormalize Min-Max
- `denormalize_unitsphere(norm_vertices, v_center, max_radius)` - Denormalize Unit Sphere
- `compute_error(original, reconstructed)` - Compute MSE/MAE errors
- `visualize_results(errors_minmax, errors_unitsphere, output_folder)` - Create plots

### Bonus Research Functions
- `rotate_mesh(vertices, angle_x, angle_y, angle_z)` - Rotate mesh
- `translate_mesh(vertices, translation)` - Translate mesh
- `compute_local_density(vertices, k=5)` - Compute vertex density
- `adaptive_quantize(vertices, local_density, base_bins=1024)` - Adaptive quantization

## Results Interpretation

### Error Metrics
- **Lower MSE/MAE** = Better reconstruction quality
- For 1024 bins, errors are typically very small (< 1e-3)
- Unit Sphere normalization may show slightly different error patterns than Min-Max

### Normalization Method Comparison
- **Min-Max**: Preserves aspect ratio, good for meshes with known bounds
- **Unit Sphere**: Rotation/translation invariant, good for shape analysis

### Adaptive Quantization
- Can improve reconstruction in dense regions
- May reduce overall error for non-uniform meshes
- Trade-off between precision and storage efficiency

## Author

Academic Assignment - Mesh Normalization, Quantization, and Error Analysis

## Version

1.0.0

## License

This project is for academic/educational purposes.

## Notes

- The script automatically creates necessary directories (`meshes/`, `output/`)
- Sample cube mesh is auto-generated if not present
- All plots are saved with 150 DPI resolution
- Error metrics are printed in scientific notation for precision

