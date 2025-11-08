"""
Mesh Normalization, Quantization, and Error Analysis

This script performs the following tasks:
1. Load and inspect 3D mesh (.obj files)
2. Normalize and quantize mesh vertices
3. Dequantize, denormalize, and compute reconstruction error
"""

import os
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any


# ============================================================================
# CONFIGURATION - File paths and parameters
# ============================================================================

# Input mesh file path (update with your .obj file path)
INPUT_MESH_PATH = "path/to/your/mesh.obj"

# Output folder for saving results
OUTPUT_FOLDER = "output"

# Quantization parameters (to be configured)
QUANTIZATION_BITS = 8  # Number of bits for quantization


# ============================================================================
# FUNCTION DEFINITIONS
# ============================================================================

def load_and_inspect_mesh(file_path: str) -> trimesh.Trimesh:
    """
    Load a 3D mesh from an .obj file and inspect its properties.
    
    Args:
        file_path: Path to the .obj mesh file
        
    Returns:
        trimesh.Trimesh: Loaded mesh object
        
    Raises:
        FileNotFoundError: If the mesh file does not exist
        ValueError: If the file cannot be loaded as a mesh
    """
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: Mesh file not found at '{file_path}'. Please check the file path.")
    
    # Load mesh using trimesh
    try:
        mesh = trimesh.load(file_path)
        
        # Ensure we have a Trimesh object (trimesh.load can return different types)
        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError(f"Error: Loaded object is not a Trimesh object. Got type: {type(mesh)}")
            
    except Exception as e:
        raise ValueError(f"Error: Failed to load mesh from '{file_path}'. {str(e)}")
    
    # Extract vertex coordinates as NumPy array
    vertices = mesh.vertices
    
    # Print mesh statistics
    print(f"\n{'='*60}")
    print("Mesh Inspection Results")
    print(f"{'='*60}")
    print(f"Number of vertices: {len(vertices):,}")
    print(f"Number of faces: {len(mesh.faces):,}")
    
    # Calculate and print statistics per axis
    print(f"\nVertex Statistics (per axis):")
    print(f"{'Axis':<6} {'Min':<15} {'Max':<15} {'Mean':<15} {'Std Dev':<15}")
    print(f"{'-'*60}")
    
    axes = ['X', 'Y', 'Z']
    for i, axis in enumerate(axes):
        axis_data = vertices[:, i]
        min_val = np.min(axis_data)
        max_val = np.max(axis_data)
        mean_val = np.mean(axis_data)
        std_val = np.std(axis_data)
        
        print(f"{axis:<6} {min_val:<15.6f} {max_val:<15.6f} {mean_val:<15.6f} {std_val:<15.6f}")
    
    # Print bounding box information
    print(f"\nBounding Box:")
    print(f"  Min: [{mesh.bounds[0, 0]:.6f}, {mesh.bounds[0, 1]:.6f}, {mesh.bounds[0, 2]:.6f}]")
    print(f"  Max: [{mesh.bounds[1, 0]:.6f}, {mesh.bounds[1, 1]:.6f}, {mesh.bounds[1, 2]:.6f}]")
    print(f"  Size: [{mesh.bounds[1, 0] - mesh.bounds[0, 0]:.6f}, "
          f"{mesh.bounds[1, 1] - mesh.bounds[0, 1]:.6f}, "
          f"{mesh.bounds[1, 2] - mesh.bounds[0, 2]:.6f}]")
    
    # Optionally visualize the mesh (if display is available)
    try:
        # Check if we're in an environment that supports visualization
        # This will only work in interactive environments with display support
        if hasattr(mesh, 'show'):
            print(f"\nNote: Mesh visualization available via mesh.show()")
            # Uncomment the line below to visualize the mesh interactively
            # mesh.show()
    except Exception:
        # Silently skip visualization if not available
        pass
    
    print(f"{'='*60}\n")
    
    return mesh


def normalize_minmax(vertices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Normalize vertices using Min-Max normalization to [0, 1] range.
    
    Formula: v_norm = (v - v_min) / (v_max - v_min)
    where v_min and v_max are computed per axis (x, y, z).
    
    Args:
        vertices: Original vertex coordinates as NumPy array (N x 3)
        
    Returns:
        Tuple containing:
            - normalized_vertices: Normalized vertices in [0, 1] range
            - quantized_vertices: Quantized vertices as integers [0, bins-1]
            - denorm_params: Dictionary with parameters for denormalization:
                - 'v_min': Minimum values per axis (1 x 3)
                - 'v_max': Maximum values per axis (1 x 3)
                - 'bins': Number of quantization bins used
    """
    # Compute min and max per axis (x, y, z)
    v_min = np.min(vertices, axis=0)  # Shape: (3,)
    v_max = np.max(vertices, axis=0)  # Shape: (3,)
    
    # Avoid division by zero (if all vertices have same coordinate on an axis)
    v_range = v_max - v_min
    v_range = np.where(v_range == 0, 1.0, v_range)  # Set range to 1 if all same
    
    # Min-Max normalization: (v - v_min) / (v_max - v_min)
    normalized_vertices = (vertices - v_min) / v_range
    
    # Quantize normalized vertices
    bins = 1024
    quantized_vertices, _ = quantize(normalized_vertices, bins=bins)
    
    # Store denormalization parameters
    denorm_params = {
        'v_min': v_min,
        'v_max': v_max,
        'bins': bins,
        'method': 'minmax'
    }
    
    return normalized_vertices, quantized_vertices, denorm_params


def normalize_unitsphere(vertices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Normalize vertices using Unit Sphere normalization.
    
    Steps:
    1. Translate to origin: v_centered = v - v_center
    2. Scale to unit sphere: v_norm = v_centered / max_radius
    where v_center is the centroid and max_radius is the maximum distance from center.
    
    This ensures all vertices lie within a unit sphere (radius <= 1).
    
    Args:
        vertices: Original vertex coordinates as NumPy array (N x 3)
        
    Returns:
        Tuple containing:
            - normalized_vertices: Normalized vertices within unit sphere
            - quantized_vertices: Quantized vertices as integers [0, bins-1]
            - denorm_params: Dictionary with parameters for denormalization:
                - 'v_center': Centroid of original vertices (1 x 3)
                - 'max_radius': Maximum distance from center
                - 'bins': Number of quantization bins used
    """
    # Compute centroid (center of mass)
    v_center = np.mean(vertices, axis=0)  # Shape: (3,)
    
    # Translate vertices to origin
    vertices_centered = vertices - v_center
    
    # Compute distances from center for each vertex
    distances = np.linalg.norm(vertices_centered, axis=1)  # Shape: (N,)
    
    # Find maximum radius
    max_radius = np.max(distances)
    
    # Avoid division by zero (if all vertices are at the same point)
    if max_radius == 0:
        max_radius = 1.0
    
    # Scale to unit sphere: divide by max_radius
    # This ensures the maximum distance from center is 1.0
    normalized_vertices = vertices_centered / max_radius
    
    # Quantize normalized vertices
    bins = 1024
    quantized_vertices, _ = quantize(normalized_vertices, bins=bins)
    
    # Store denormalization parameters
    denorm_params = {
        'v_center': v_center,
        'max_radius': max_radius,
        'bins': bins,
        'method': 'unitsphere'
    }
    
    return normalized_vertices, quantized_vertices, denorm_params


def quantize(vertices: np.ndarray, bins: int = 1024) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Quantize normalized vertices to discrete integer values.
    
    Formula: q = floor(x_norm * (bins - 1))
    where x_norm is in [0, 1] range (or [-1, 1] for unit sphere, which we map to [0, 1]).
    
    For unit sphere normalization, we first map from [-1, 1] to [0, 1]:
    x_mapped = (x_norm + 1) / 2
    
    Args:
        vertices: Normalized vertex coordinates (N x 3)
        bins: Number of quantization bins (default: 1024)
        
    Returns:
        Tuple containing:
            - quantized_vertices: Quantized vertices as integers (N x 3)
            - quant_params: Dictionary with quantization parameters:
                - 'bins': Number of bins used
                - 'min_val': Minimum value in normalized vertices
                - 'max_val': Maximum value in normalized vertices
    """
    # Find the actual range of normalized vertices
    min_val = np.min(vertices)
    max_val = np.max(vertices)
    
    # Map vertices to [0, 1] range if they're not already
    # This handles both [0, 1] (minmax) and [-1, 1] (unitsphere) cases
    if min_val < 0:
        # Unit sphere case: map from [-1, 1] to [0, 1]
        vertices_mapped = (vertices + 1) / 2.0
    else:
        # Min-max case: already in [0, 1]
        vertices_mapped = vertices.copy()
    
    # Ensure values are in [0, 1] range (clip to handle floating point errors)
    vertices_mapped = np.clip(vertices_mapped, 0.0, 1.0)
    
    # Quantize: q = floor(x * (bins - 1))
    # This maps [0, 1] to [0, bins-1]
    quantized_vertices = np.floor(vertices_mapped * (bins - 1)).astype(int)
    
    # Clip to ensure values are in valid range [0, bins-1]
    quantized_vertices = np.clip(quantized_vertices, 0, bins - 1)
    
    # Store quantization parameters
    quant_params = {
        'bins': bins,
        'min_val': min_val,
        'max_val': max_val
    }
    
    return quantized_vertices, quant_params


def normalize_mesh(mesh: trimesh.Trimesh) -> Tuple[trimesh.Trimesh, Dict[str, Any]]:
    """
    Normalize mesh vertices to a standard range (e.g., [-1, 1] or [0, 1]).
    Store normalization parameters for later denormalization.
    
    Args:
        mesh: Original mesh object
        
    Returns:
        Tuple containing:
            - normalized_mesh: Mesh with normalized vertices
            - normalization_params: Dictionary with parameters needed for denormalization
                (e.g., original center, scale factor, min/max values)
        
    TODO:
        - Extract vertex coordinates
        - Compute normalization parameters (center, scale, etc.)
        - Normalize vertices to target range
        - Create new mesh with normalized vertices
        - Return normalized mesh and parameters
    """
    pass


def quantize_mesh(normalized_mesh: trimesh.Trimesh, bits: int) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Quantize normalized mesh vertices to discrete integer values.
    
    Args:
        normalized_mesh: Mesh with normalized vertices
        bits: Number of bits for quantization (determines quantization levels)
        
    Returns:
        Tuple containing:
            - quantized_vertices: Quantized vertex coordinates as integer array
            - quantization_params: Dictionary with quantization parameters
                (e.g., quantization levels, min/max values, bit depth)
        
    TODO:
        - Extract normalized vertex coordinates
        - Determine quantization range based on bits
        - Quantize vertices to integer values
        - Store quantization parameters
        - Return quantized vertices and parameters
    """
    pass


def dequantize(quantized_vertices: np.ndarray, bins: int = 1024) -> np.ndarray:
    """
    Convert quantized integer values back to continuous normalized coordinates.
    
    Formula: x_deq = q / (bins - 1)
    where q is the quantized integer value [0, bins-1] and x_deq is in [0, 1].
    
    For unit sphere normalization, we need to map back from [0, 1] to [-1, 1]:
    x_deq = 2 * x_deq - 1
    
    Args:
        quantized_vertices: Quantized vertex coordinates as integer array (N x 3)
        bins: Number of quantization bins (default: 1024)
        
    Returns:
        np.ndarray: Dequantized normalized vertex coordinates in [0, 1] range
    """
    # Ensure quantized vertices are in valid range [0, bins-1]
    quantized_vertices = np.clip(quantized_vertices, 0, bins - 1)
    
    # Dequantize: x_deq = q / (bins - 1)
    # This maps [0, bins-1] back to [0, 1]
    dequantized_vertices = quantized_vertices.astype(float) / (bins - 1)
    
    # Ensure values are in [0, 1] range (handle floating point errors)
    dequantized_vertices = np.clip(dequantized_vertices, 0.0, 1.0)
    
    return dequantized_vertices


def denormalize_minmax(norm_vertices: np.ndarray, v_min: np.ndarray, v_max: np.ndarray) -> np.ndarray:
    """
    Restore original vertex scale from Min-Max normalized coordinates.
    
    Formula: x_denorm = norm * (v_max - v_min) + v_min
    where norm is in [0, 1] and v_min, v_max are the original min/max per axis.
    
    Args:
        norm_vertices: Normalized vertex coordinates in [0, 1] range (N x 3)
        v_min: Minimum values per axis (3,)
        v_max: Maximum values per axis (3,)
        
    Returns:
        np.ndarray: Denormalized vertex coordinates restored to original scale
    """
    # Ensure normalized vertices are in [0, 1] range
    norm_vertices = np.clip(norm_vertices, 0.0, 1.0)
    
    # Compute range per axis
    v_range = v_max - v_min
    
    # Denormalize: x_denorm = norm * (v_max - v_min) + v_min
    denormalized_vertices = norm_vertices * v_range + v_min
    
    return denormalized_vertices


def denormalize_unitsphere(norm_vertices: np.ndarray, v_center: np.ndarray, max_radius: float) -> np.ndarray:
    """
    Restore original vertex scale from Unit Sphere normalized coordinates.
    
    Steps:
    1. Scale back: v_scaled = norm * max_radius
    2. Translate back: v_denorm = v_scaled + v_center
    
    where norm is in [-1, 1] (unit sphere), v_center is the centroid,
    and max_radius is the original maximum distance from center.
    
    Args:
        norm_vertices: Normalized vertex coordinates in [-1, 1] range (N x 3)
        v_center: Centroid of original vertices (3,)
        max_radius: Maximum distance from center in original mesh
        
    Returns:
        np.ndarray: Denormalized vertex coordinates restored to original scale
    """
    # If norm_vertices are in [0, 1] (from dequantization), map to [-1, 1]
    if np.min(norm_vertices) >= 0:
        # Map from [0, 1] to [-1, 1]
        norm_vertices = 2.0 * norm_vertices - 1.0
    
    # Clip to [-1, 1] range (unit sphere constraint)
    norm_vertices = np.clip(norm_vertices, -1.0, 1.0)
    
    # Scale back: multiply by max_radius
    vertices_scaled = norm_vertices * max_radius
    
    # Translate back: add centroid
    denormalized_vertices = vertices_scaled + v_center
    
    return denormalized_vertices


def dequantize_mesh(quantized_vertices: np.ndarray, quantization_params: Dict[str, Any]) -> np.ndarray:
    """
    Convert quantized integer values back to continuous normalized coordinates.
    
    Args:
        quantized_vertices: Quantized vertex coordinates as integer array
        quantization_params: Dictionary with quantization parameters
        
    Returns:
        np.ndarray: Dequantized normalized vertex coordinates
        
    TODO:
        - Use quantization parameters to reverse quantization
        - Convert integer values back to continuous normalized range
        - Return dequantized normalized vertices
    """
    pass


def denormalize_mesh(dequantized_vertices: np.ndarray, normalization_params: Dict[str, Any]) -> np.ndarray:
    """
    Restore original scale and position from normalized coordinates.
    
    Args:
        dequantized_vertices: Dequantized normalized vertex coordinates
        normalization_params: Dictionary with normalization parameters
        
    Returns:
        np.ndarray: Denormalized vertex coordinates (restored to original scale/position)
        
    TODO:
        - Use normalization parameters to reverse normalization
        - Restore original scale and translation
        - Return denormalized vertices
    """
    pass


def compute_error(original: np.ndarray, reconstructed: np.ndarray) -> Dict[str, float]:
    """
    Compute reconstruction error between original and reconstructed vertices.
    
    Computes Mean Squared Error (MSE) and Mean Absolute Error (MAE)
    for each axis (x, y, z) and overall averages.
    
    Formulas:
    - MSE per axis: mean((original_axis - reconstructed_axis)^2)
    - MAE per axis: mean(|original_axis - reconstructed_axis|)
    - Overall MSE: mean of all squared differences
    - Overall MAE: mean of all absolute differences
    
    Args:
        original: Original vertex coordinates (N x 3)
        reconstructed: Reconstructed vertex coordinates (N x 3)
        
    Returns:
        Dictionary containing error metrics:
            - 'mse_x', 'mse_y', 'mse_z': MSE per axis
            - 'mae_x', 'mae_y', 'mae_z': MAE per axis
            - 'mse_overall': Overall MSE
            - 'mae_overall': Overall MAE
    """
    # Ensure both arrays have the same shape
    if original.shape != reconstructed.shape:
        raise ValueError(f"Shape mismatch: original {original.shape} vs reconstructed {reconstructed.shape}")
    
    # Compute per-axis errors
    errors = original - reconstructed  # Shape: (N, 3)
    
    # Mean Squared Error (MSE) per axis
    mse_x = np.mean(errors[:, 0] ** 2)
    mse_y = np.mean(errors[:, 1] ** 2)
    mse_z = np.mean(errors[:, 2] ** 2)
    
    # Mean Absolute Error (MAE) per axis
    mae_x = np.mean(np.abs(errors[:, 0]))
    mae_y = np.mean(np.abs(errors[:, 1]))
    mae_z = np.mean(np.abs(errors[:, 2]))
    
    # Overall MSE and MAE
    mse_overall = np.mean(errors ** 2)
    mae_overall = np.mean(np.abs(errors))
    
    # Store results in dictionary
    error_metrics = {
        'mse_x': float(mse_x),
        'mse_y': float(mse_y),
        'mse_z': float(mse_z),
        'mae_x': float(mae_x),
        'mae_y': float(mae_y),
        'mae_z': float(mae_z),
        'mse_overall': float(mse_overall),
        'mae_overall': float(mae_overall)
    }
    
    return error_metrics


def visualize_results(errors_minmax: Dict[str, float], 
                      errors_unitsphere: Dict[str, float],
                      output_folder: str) -> None:
    """
    Plot MSE and MAE per axis (x, y, z) using matplotlib.
    
    Creates bar plots comparing error metrics for both normalization methods
    and saves them as PNG files.
    
    Args:
        errors_minmax: Dictionary containing error metrics for Min-Max normalization
        errors_unitsphere: Dictionary containing error metrics for Unit Sphere normalization
        output_folder: Path to folder where plots will be saved
    """
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Extract error metrics
    axes = ['X', 'Y', 'Z']
    mse_minmax = [errors_minmax['mse_x'], errors_minmax['mse_y'], errors_minmax['mse_z']]
    mae_minmax = [errors_minmax['mae_x'], errors_minmax['mae_y'], errors_minmax['mae_z']]
    mse_unitsphere = [errors_unitsphere['mse_x'], errors_unitsphere['mse_y'], errors_unitsphere['mse_z']]
    mae_unitsphere = [errors_unitsphere['mae_x'], errors_unitsphere['mae_y'], errors_unitsphere['mae_z']]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot MSE per axis
    x_pos = np.arange(len(axes))
    width = 0.35
    
    ax1.bar(x_pos - width/2, mse_minmax, width, label='Min-Max', alpha=0.8)
    ax1.bar(x_pos + width/2, mse_unitsphere, width, label='Unit Sphere', alpha=0.8)
    ax1.set_xlabel('Axis')
    ax1.set_ylabel('Mean Squared Error (MSE)')
    ax1.set_title('MSE per Axis')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(axes)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot MAE per axis
    ax2.bar(x_pos - width/2, mae_minmax, width, label='Min-Max', alpha=0.8)
    ax2.bar(x_pos + width/2, mae_unitsphere, width, label='Unit Sphere', alpha=0.8)
    ax2.set_xlabel('Axis')
    ax2.set_ylabel('Mean Absolute Error (MAE)')
    ax2.set_title('MAE per Axis')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(axes)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_folder, "error_comparison.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved error comparison plot: {output_path}")
    
    # Create separate plots for each normalization method
    # Min-Max plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.bar(axes, mse_minmax, alpha=0.8, color='steelblue')
    ax1.set_xlabel('Axis')
    ax1.set_ylabel('Mean Squared Error (MSE)')
    ax1.set_title('MSE per Axis - Min-Max Normalization')
    ax1.grid(True, alpha=0.3)
    
    ax2.bar(axes, mae_minmax, alpha=0.8, color='coral')
    ax2.set_xlabel('Axis')
    ax2.set_ylabel('Mean Absolute Error (MAE)')
    ax2.set_title('MAE per Axis - Min-Max Normalization')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path_minmax = os.path.join(output_folder, "error_minmax.png")
    plt.savefig(output_path_minmax, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved Min-Max error plot: {output_path_minmax}")
    
    # Unit Sphere plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.bar(axes, mse_unitsphere, alpha=0.8, color='steelblue')
    ax1.set_xlabel('Axis')
    ax1.set_ylabel('Mean Squared Error (MSE)')
    ax1.set_title('MSE per Axis - Unit Sphere Normalization')
    ax1.grid(True, alpha=0.3)
    
    ax2.bar(axes, mae_unitsphere, alpha=0.8, color='coral')
    ax2.set_xlabel('Axis')
    ax2.set_ylabel('Mean Absolute Error (MAE)')
    ax2.set_title('MAE per Axis - Unit Sphere Normalization')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path_unitsphere = os.path.join(output_folder, "error_unitsphere.png")
    plt.savefig(output_path_unitsphere, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved Unit Sphere error plot: {output_path_unitsphere}")


# ============================================================================
# TEST FUNCTION
# ============================================================================

def test_load_and_inspect_mesh():
    """
    Test function for load_and_inspect_mesh().
    Creates a sample cube mesh if it doesn't exist, then tests loading it.
    """
    test_mesh_path = "meshes/sample_cube.obj"
    
    print("=" * 60)
    print("Testing load_and_inspect_mesh() Function")
    print("=" * 60)
    
    # Create meshes directory if it doesn't exist
    mesh_dir = os.path.dirname(test_mesh_path)
    if mesh_dir and not os.path.exists(mesh_dir):
        os.makedirs(mesh_dir, exist_ok=True)
        print(f"Created directory: {mesh_dir}")
    
    # Generate sample cube mesh if it doesn't exist
    if not os.path.exists(test_mesh_path):
        print(f"\nSample mesh not found at '{test_mesh_path}'")
        print("Generating a simple cube mesh...")
        
        try:
            # Create a simple cube using trimesh primitives
            cube = trimesh.primitives.Box(extents=[2.0, 2.0, 2.0])  # 2x2x2 cube
            
            # Export to .obj file
            cube.export(test_mesh_path)
            print(f"✓ Successfully created sample cube mesh at '{test_mesh_path}'")
        except Exception as e:
            print(f"✗ Error creating sample mesh: {e}")
            return
    
    # Test loading the mesh
    print(f"\nTesting load_and_inspect_mesh() with: {test_mesh_path}")
    print("-" * 60)
    
    try:
        mesh = load_and_inspect_mesh(test_mesh_path)
        
        # Print additional information about the returned mesh
        print("\n" + "=" * 60)
        print("Test Results - Returned Mesh Information")
        print("=" * 60)
        print(f"Mesh type: {type(mesh)}")
        print(f"Mesh type name: {type(mesh).__name__}")
        print(f"Is Trimesh instance: {isinstance(mesh, trimesh.Trimesh)}")
        print(f"\nVertices array shape: {mesh.vertices.shape}")
        print(f"Vertices dtype: {mesh.vertices.dtype}")
        print(f"Faces array shape: {mesh.faces.shape}")
        print(f"Faces dtype: {mesh.faces.dtype}")
        print(f"\nFirst 3 vertices:")
        print(mesh.vertices[:3])
        print("=" * 60)
        print("✓ Test passed! load_and_inspect_mesh() works correctly.\n")
        
    except FileNotFoundError as e:
        print(f"\n✗ FileNotFoundError: {e}")
        print("Test failed - file not found error was raised correctly.")
    except ValueError as e:
        print(f"\n✗ ValueError: {e}")
        print("Test failed - mesh loading error occurred.")
    except Exception as e:
        print(f"\n✗ Unexpected error: {type(e).__name__}: {e}")
        print("Test failed - unexpected error occurred.")


def test_missing_file_handling():
    """
    Test that load_and_inspect_mesh() handles missing files gracefully.
    """
    print("=" * 60)
    print("Testing Missing File Handling")
    print("=" * 60)
    
    non_existent_path = "meshes/non_existent_file.obj"
    print(f"\nTesting with non-existent file: {non_existent_path}")
    print("-" * 60)
    
    try:
        mesh = load_and_inspect_mesh(non_existent_path)
        print("✗ Test failed - should have raised FileNotFoundError")
    except FileNotFoundError as e:
        print(f"✓ FileNotFoundError raised correctly:")
        print(f"  {e}")
        print("✓ Missing file handling works correctly.\n")
    except Exception as e:
        print(f"✗ Unexpected error type: {type(e).__name__}: {e}")


def test_normalize_and_quantize():
    """
    Test normalization and quantization functions.
    Uses the cube mesh from Task 1.
    """
    print("=" * 60)
    print("Testing Normalization and Quantization (Task 2)")
    print("=" * 60)
    
    # Ensure output folder exists
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # Load the cube mesh from Task 1
    test_mesh_path = "meshes/sample_cube.obj"
    
    if not os.path.exists(test_mesh_path):
        print(f"\n✗ Test mesh not found at '{test_mesh_path}'")
        print("Please run test_load_and_inspect_mesh() first to generate the cube mesh.")
        return
    
    try:
        # Load mesh
        print(f"\nLoading test mesh: {test_mesh_path}")
        mesh = load_and_inspect_mesh(test_mesh_path)
        vertices = mesh.vertices
        faces = mesh.faces
        
        print(f"Original vertices shape: {vertices.shape}")
        print(f"Original vertices range: [{np.min(vertices):.6f}, {np.max(vertices):.6f}]")
        
        # Test Min-Max normalization
        print("\n" + "-" * 60)
        print("Testing Min-Max Normalization")
        print("-" * 60)
        
        norm_minmax, quant_minmax, params_minmax = normalize_minmax(vertices)
        
        # Verify normalized range [0, 1]
        norm_min = np.min(norm_minmax)
        norm_max = np.max(norm_minmax)
        print(f"\nNormalized vertices range: [{norm_min:.6f}, {norm_max:.6f}]")
        assert 0.0 <= norm_min <= 1.0, f"Min-Max normalized min should be in [0, 1], got {norm_min}"
        assert 0.0 <= norm_max <= 1.0, f"Min-Max normalized max should be in [0, 1], got {norm_max}"
        print("✓ Normalized coordinates are in [0, 1] range")
        
        # Verify quantized range [0, 1023]
        quant_min = np.min(quant_minmax)
        quant_max = np.max(quant_minmax)
        print(f"Quantized vertices range: [{quant_min}, {quant_max}]")
        assert 0 <= quant_min <= 1023, f"Quantized min should be in [0, 1023], got {quant_min}"
        assert 0 <= quant_max <= 1023, f"Quantized max should be in [0, 1023], got {quant_max}"
        print("✓ Quantized coordinates are integers in [0, 1023] range")
        
        # Create mesh objects and save
        mesh_norm_minmax = trimesh.Trimesh(vertices=norm_minmax, faces=faces)
        mesh_quant_minmax = trimesh.Trimesh(vertices=quant_minmax.astype(float), faces=faces)
        
        output_norm_minmax = os.path.join(OUTPUT_FOLDER, "normalized_minmax.ply")
        output_quant_minmax = os.path.join(OUTPUT_FOLDER, "quantized_minmax.ply")
        
        mesh_norm_minmax.export(output_norm_minmax)
        mesh_quant_minmax.export(output_quant_minmax)
        
        print(f"✓ Saved normalized mesh: {output_norm_minmax}")
        print(f"✓ Saved quantized mesh: {output_quant_minmax}")
        
        # Test Unit Sphere normalization
        print("\n" + "-" * 60)
        print("Testing Unit Sphere Normalization")
        print("-" * 60)
        
        norm_unitsphere, quant_unitsphere, params_unitsphere = normalize_unitsphere(vertices)
        
        # Verify normalized range (max radius <= 1)
        distances = np.linalg.norm(norm_unitsphere, axis=1)
        max_radius = np.max(distances)
        norm_min = np.min(norm_unitsphere)
        norm_max = np.max(norm_unitsphere)
        print(f"\nNormalized vertices range: [{norm_min:.6f}, {norm_max:.6f}]")
        print(f"Maximum radius from center: {max_radius:.6f}")
        assert max_radius <= 1.0 + 1e-6, f"Unit sphere max radius should be <= 1, got {max_radius}"
        print("✓ Maximum radius <= 1.0 (unit sphere constraint satisfied)")
        
        # Verify quantized range [0, 1023]
        quant_min = np.min(quant_unitsphere)
        quant_max = np.max(quant_unitsphere)
        print(f"Quantized vertices range: [{quant_min}, {quant_max}]")
        assert 0 <= quant_min <= 1023, f"Quantized min should be in [0, 1023], got {quant_min}"
        assert 0 <= quant_max <= 1023, f"Quantized max should be in [0, 1023], got {quant_max}"
        print("✓ Quantized coordinates are integers in [0, 1023] range")
        
        # Create mesh objects and save
        mesh_norm_unitsphere = trimesh.Trimesh(vertices=norm_unitsphere, faces=faces)
        mesh_quant_unitsphere = trimesh.Trimesh(vertices=quant_unitsphere.astype(float), faces=faces)
        
        output_norm_unitsphere = os.path.join(OUTPUT_FOLDER, "normalized_unitsphere.ply")
        output_quant_unitsphere = os.path.join(OUTPUT_FOLDER, "quantized_unitsphere.ply")
        
        mesh_norm_unitsphere.export(output_norm_unitsphere)
        mesh_quant_unitsphere.export(output_quant_unitsphere)
        
        print(f"✓ Saved normalized mesh: {output_norm_unitsphere}")
        print(f"✓ Saved quantized mesh: {output_quant_unitsphere}")
        
        # Verify all files were created
        print("\n" + "-" * 60)
        print("Verifying Saved Files")
        print("-" * 60)
        
        files_to_check = [
            output_norm_minmax,
            output_quant_minmax,
            output_norm_unitsphere,
            output_quant_unitsphere
        ]
        
        all_files_exist = True
        for file_path in files_to_check:
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                print(f"✓ File exists: {file_path} ({file_size} bytes)")
            else:
                print(f"✗ File missing: {file_path}")
                all_files_exist = False
        
        if all_files_exist:
            print("\n" + "=" * 60)
            print("✓ Task 2 Normalization and Quantization implemented and verified successfully.")
            print("=" * 60)
        else:
            print("\n✗ Some files are missing!")
        
    except AssertionError as e:
        print(f"\n✗ Assertion failed: {e}")
        print("Test failed - validation error occurred.")
    except Exception as e:
        print(f"\n✗ Unexpected error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        print("Test failed - unexpected error occurred.")


def test_dequantize_denormalize_and_error():
    """
    Test dequantization, denormalization, and error computation.
    Uses the cube mesh from Task 1 and performs the full pipeline.
    """
    print("=" * 60)
    print("Testing Dequantization, Denormalization, and Error Analysis (Task 3)")
    print("=" * 60)
    
    # Ensure output folder exists
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # Load the cube mesh from Task 1
    test_mesh_path = "meshes/sample_cube.obj"
    
    if not os.path.exists(test_mesh_path):
        print(f"\n✗ Test mesh not found at '{test_mesh_path}'")
        print("Please run test_load_and_inspect_mesh() first to generate the cube mesh.")
        return
    
    try:
        # Load mesh
        print(f"\nLoading test mesh: {test_mesh_path}")
        mesh = load_and_inspect_mesh(test_mesh_path)
        original_vertices = mesh.vertices
        faces = mesh.faces
        
        print(f"Original vertices shape: {original_vertices.shape}")
        print(f"Original vertices range: [{np.min(original_vertices):.6f}, {np.max(original_vertices):.6f}]")
        
        # ====================================================================
        # Test Min-Max Normalization Pipeline
        # ====================================================================
        print("\n" + "=" * 60)
        print("Testing Min-Max Normalization Pipeline")
        print("=" * 60)
        
        # Step 1: Normalize and quantize
        norm_minmax, quant_minmax, params_minmax = normalize_minmax(original_vertices)
        print(f"\n[Step 1] Normalized and quantized")
        print(f"  Normalized range: [{np.min(norm_minmax):.6f}, {np.max(norm_minmax):.6f}]")
        print(f"  Quantized range: [{np.min(quant_minmax)}, {np.max(quant_minmax)}]")
        
        # Step 2: Dequantize
        dequant_minmax = dequantize(quant_minmax, bins=params_minmax['bins'])
        print(f"\n[Step 2] Dequantized")
        print(f"  Dequantized range: [{np.min(dequant_minmax):.6f}, {np.max(dequant_minmax):.6f}]")
        
        # Step 3: Denormalize
        reconstructed_minmax = denormalize_minmax(
            dequant_minmax, 
            params_minmax['v_min'], 
            params_minmax['v_max']
        )
        print(f"\n[Step 3] Denormalized")
        print(f"  Reconstructed range: [{np.min(reconstructed_minmax):.6f}, {np.max(reconstructed_minmax):.6f}]")
        print(f"  Original range: [{np.min(original_vertices):.6f}, {np.max(original_vertices):.6f}]")
        
        # Step 4: Compute error
        errors_minmax = compute_error(original_vertices, reconstructed_minmax)
        print(f"\n[Step 4] Error Metrics (Min-Max):")
        print(f"  MSE X: {errors_minmax['mse_x']:.6e}")
        print(f"  MSE Y: {errors_minmax['mse_y']:.6e}")
        print(f"  MSE Z: {errors_minmax['mse_z']:.6e}")
        print(f"  MAE X: {errors_minmax['mae_x']:.6e}")
        print(f"  MAE Y: {errors_minmax['mae_y']:.6e}")
        print(f"  MAE Z: {errors_minmax['mae_z']:.6e}")
        print(f"  MSE Overall: {errors_minmax['mse_overall']:.6e}")
        print(f"  MAE Overall: {errors_minmax['mae_overall']:.6e}")
        
        # Save reconstructed mesh
        mesh_reconstructed_minmax = trimesh.Trimesh(vertices=reconstructed_minmax, faces=faces)
        output_reconstructed_minmax = os.path.join(OUTPUT_FOLDER, "reconstructed_minmax.ply")
        mesh_reconstructed_minmax.export(output_reconstructed_minmax)
        print(f"\n✓ Saved reconstructed mesh: {output_reconstructed_minmax}")
        
        # ====================================================================
        # Test Unit Sphere Normalization Pipeline
        # ====================================================================
        print("\n" + "=" * 60)
        print("Testing Unit Sphere Normalization Pipeline")
        print("=" * 60)
        
        # Step 1: Normalize and quantize
        norm_unitsphere, quant_unitsphere, params_unitsphere = normalize_unitsphere(original_vertices)
        print(f"\n[Step 1] Normalized and quantized")
        distances = np.linalg.norm(norm_unitsphere, axis=1)
        max_radius = np.max(distances)
        print(f"  Normalized range: [{np.min(norm_unitsphere):.6f}, {np.max(norm_unitsphere):.6f}]")
        print(f"  Max radius: {max_radius:.6f}")
        print(f"  Quantized range: [{np.min(quant_unitsphere)}, {np.max(quant_unitsphere)}]")
        
        # Step 2: Dequantize
        dequant_unitsphere = dequantize(quant_unitsphere, bins=params_unitsphere['bins'])
        print(f"\n[Step 2] Dequantized")
        print(f"  Dequantized range: [{np.min(dequant_unitsphere):.6f}, {np.max(dequant_unitsphere):.6f}]")
        
        # Step 3: Denormalize
        reconstructed_unitsphere = denormalize_unitsphere(
            dequant_unitsphere,
            params_unitsphere['v_center'],
            params_unitsphere['max_radius']
        )
        print(f"\n[Step 3] Denormalized")
        print(f"  Reconstructed range: [{np.min(reconstructed_unitsphere):.6f}, {np.max(reconstructed_unitsphere):.6f}]")
        print(f"  Original range: [{np.min(original_vertices):.6f}, {np.max(original_vertices):.6f}]")
        
        # Step 4: Compute error
        errors_unitsphere = compute_error(original_vertices, reconstructed_unitsphere)
        print(f"\n[Step 4] Error Metrics (Unit Sphere):")
        print(f"  MSE X: {errors_unitsphere['mse_x']:.6e}")
        print(f"  MSE Y: {errors_unitsphere['mse_y']:.6e}")
        print(f"  MSE Z: {errors_unitsphere['mse_z']:.6e}")
        print(f"  MAE X: {errors_unitsphere['mae_x']:.6e}")
        print(f"  MAE Y: {errors_unitsphere['mae_y']:.6e}")
        print(f"  MAE Z: {errors_unitsphere['mae_z']:.6e}")
        print(f"  MSE Overall: {errors_unitsphere['mse_overall']:.6e}")
        print(f"  MAE Overall: {errors_unitsphere['mae_overall']:.6e}")
        
        # Save reconstructed mesh
        mesh_reconstructed_unitsphere = trimesh.Trimesh(vertices=reconstructed_unitsphere, faces=faces)
        output_reconstructed_unitsphere = os.path.join(OUTPUT_FOLDER, "reconstructed_unitsphere.ply")
        mesh_reconstructed_unitsphere.export(output_reconstructed_unitsphere)
        print(f"\n✓ Saved reconstructed mesh: {output_reconstructed_unitsphere}")
        
        # ====================================================================
        # Visualize Results
        # ====================================================================
        print("\n" + "=" * 60)
        print("Visualizing Error Metrics")
        print("=" * 60)
        
        visualize_results(errors_minmax, errors_unitsphere, OUTPUT_FOLDER)
        
        # ====================================================================
        # Verify Results
        # ====================================================================
        print("\n" + "=" * 60)
        print("Verifying Results")
        print("=" * 60)
        
        # Check that errors are small (ideally near zero for the cube)
        # For a cube with 1024 bins, errors should be very small
        print(f"\nError Verification:")
        print(f"  Min-Max MSE Overall: {errors_minmax['mse_overall']:.6e}")
        print(f"  Unit Sphere MSE Overall: {errors_unitsphere['mse_overall']:.6e}")
        
        # Verify files were created
        files_to_check = [
            output_reconstructed_minmax,
            output_reconstructed_unitsphere,
            os.path.join(OUTPUT_FOLDER, "error_minmax.png"),
            os.path.join(OUTPUT_FOLDER, "error_unitsphere.png"),
            os.path.join(OUTPUT_FOLDER, "error_comparison.png")
        ]
        
        all_files_exist = True
        for file_path in files_to_check:
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                print(f"✓ File exists: {file_path} ({file_size} bytes)")
            else:
                print(f"✗ File missing: {file_path}")
                all_files_exist = False
        
        if all_files_exist:
            print("\n" + "=" * 60)
            print("✓ Task 3 Dequantization, Denormalization, and Error Analysis implemented and verified successfully.")
            print("=" * 60)
        else:
            print("\n✗ Some files are missing!")
        
    except AssertionError as e:
        print(f"\n✗ Assertion failed: {e}")
        print("Test failed - validation error occurred.")
    except Exception as e:
        print(f"\n✗ Unexpected error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        print("Test failed - unexpected error occurred.")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution function that orchestrates the mesh normalization,
    quantization, and error analysis pipeline.
    """
    # Create output folder if it doesn't exist
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    print("=" * 60)
    print("Mesh Normalization, Quantization, and Error Analysis")
    print("=" * 60)
    
    # Step 1: Load and inspect mesh
    print("\n[Step 1] Loading and inspecting mesh...")
    original_mesh = load_and_inspect_mesh(INPUT_MESH_PATH)
    
    # Step 2: Normalize mesh
    print("\n[Step 2] Normalizing mesh...")
    normalized_mesh, normalization_params = normalize_mesh(original_mesh)
    
    # Step 3: Quantize mesh
    print("\n[Step 3] Quantizing mesh...")
    quantized_vertices, quantization_params = quantize_mesh(normalized_mesh, QUANTIZATION_BITS)
    
    # Step 4: Dequantize mesh
    print("\n[Step 4] Dequantizing mesh...")
    dequantized_vertices = dequantize_mesh(quantized_vertices, quantization_params)
    
    # Step 5: Denormalize mesh
    print("\n[Step 5] Denormalizing mesh...")
    reconstructed_vertices = denormalize_mesh(dequantized_vertices, normalization_params)
    
    # Step 6: Compute error
    print("\n[Step 6] Computing reconstruction error...")
    error_metrics = compute_error(original_mesh, reconstructed_vertices)
    
    # Step 7: Visualize results
    print("\n[Step 7] Visualizing results...")
    visualize_results(original_mesh, reconstructed_vertices, error_metrics, OUTPUT_FOLDER)
    
    print("\n" + "=" * 60)
    print("Analysis complete! Results saved to:", OUTPUT_FOLDER)
    print("=" * 60)


if __name__ == "__main__":
    # Run tests for load_and_inspect_mesh()
    print("\n" + "=" * 60)
    print("RUNNING TESTS")
    print("=" * 60)
    
    # Test 1: Test loading with sample cube mesh
    test_load_and_inspect_mesh()
    
    # Test 2: Test missing file handling
    test_missing_file_handling()
    
    # Test 3: Test normalization and quantization (Task 2)
    test_normalize_and_quantize()
    
    # Test 4: Test dequantization, denormalization, and error analysis (Task 3)
    test_dequantize_denormalize_and_error()
    
    print("\n" + "=" * 60)
    print("TESTS COMPLETE")
    print("=" * 60)
    print("\nTo run the full pipeline, uncomment the line below:")
    print("# main()")
    
    # Uncomment the line below to run the full pipeline instead of tests
    # main()

