import cv2
import numpy as np
import os

def draw_marker_on_resized_image_no_transform(image_path, x, y, target_width=874, target_height=600, output_path=None):
    """
    Resize image and draw a marker at the same coordinates (no transformation)
    
    Args:
        image_path (str): Path to the input image
        x (int): X coordinate (unchanged)
        y (int): Y coordinate (unchanged)
        target_width (int): Target width for resized image
        target_height (int): Target height for resized image
        output_path (str): Path to save the output image (optional)
    """
    
    # Load the image
    original_image = cv2.imread(image_path)
    
    if original_image is None:
        print(f"Error: Could not load image from {image_path}")
        return None
    
    # Get original image dimensions
    original_height, original_width = original_image.shape[:2]
    print(f"Original image dimensions: {original_width}x{original_height}")
    
    # Resize the image
    resized_image = cv2.resize(original_image, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    print(f"Resized image dimensions: {target_width}x{target_height}")
    
    # Make a copy to preserve the resized image
    marked_image = resized_image.copy()
    
    # Check if coordinates are within bounds of resized image
    if x >= target_width or y >= target_height or x < 0 or y < 0:
        print(f"Warning: Coordinates ({x}, {y}) are outside the resized image bounds ({target_width}x{target_height})")
        print("Marker may not be visible or may cause errors")
    else:
        print(f"Coordinates ({x}, {y}) are within bounds of resized image")
    
    # Define marker properties
    marker_color = (0, 0, 255)  # Red color in BGR format
    marker_size = 8
    thickness = 2
    
    # Draw a circle marker at the same coordinates
    cv2.circle(marked_image, (x, y), marker_size, marker_color, thickness)
    
    # Draw crosshairs for better visibility
    crosshair_length = 12
    cv2.line(marked_image, 
             (x - crosshair_length, y), 
             (x + crosshair_length, y), 
             marker_color, thickness)
    cv2.line(marked_image, 
             (x, y - crosshair_length), 
             (x, y + crosshair_length), 
             marker_color, thickness)
    
    # Add coordinate text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    text_color = (0, 255, 0)  # Green color for text
    text_thickness = 1
    
    # Position text slightly offset from the marker
    text = f"({x},{y})"
    
    # Calculate text position to avoid going outside image bounds
    text_x = min(x + 15, target_width - 80)
    text_y = max(y - 15, 20)
    
    cv2.putText(marked_image, text, (text_x, text_y), 
                font, font_scale, text_color, text_thickness)
    
    # Generate output path if not provided
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_dir = os.path.dirname(image_path)
        output_path = os.path.join(output_dir, f"{base_name}_resized_{target_width}x{target_height}_same_coords.png")
    
    # Save the marked resized image
    cv2.imwrite(output_path, marked_image)
    print(f"Marked resized image saved to: {output_path}")
    
    # Also save just the resized image without marker
    resized_only_path = os.path.join(os.path.dirname(output_path), 
                                   f"{os.path.splitext(os.path.basename(output_path))[0]}_no_marker.png")
    cv2.imwrite(resized_only_path, resized_image)
    print(f"Resized image (no marker) saved to: {resized_only_path}")
    
    return marked_image

def draw_marker_on_image(image_path, x, y, output_path=None):
    """
    Draw a marker/point on specified coordinates of an image
    
    Args:
        image_path (str): Path to the input image
        x (int): X coordinate
        y (int): Y coordinate
        output_path (str): Path to save the output image (optional)
    """
    
    # Load the image
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    # Make a copy to preserve the original
    marked_image = image.copy()
    
    # Define marker properties
    marker_color = (0, 0, 255)  # Red color in BGR format
    marker_size = 10
    thickness = 2
    
    # Draw a circle marker
    cv2.circle(marked_image, (x, y), marker_size, marker_color, thickness)
    
    # Optional: Draw crosshairs for better visibility
    cv2.line(marked_image, (x - 15, y), (x + 15, y), marker_color, thickness)
    cv2.line(marked_image, (x, y - 15), (x, y + 15), marker_color, thickness)
    
    # Add coordinate text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    text_color = (0, 255, 0)  # Green color for text
    text_thickness = 2
    
    # Position text slightly offset from the marker
    text = f"({x},{y})"
    text_x = x + 20
    text_y = y - 20
    
    cv2.putText(marked_image, text, (text_x, text_y), font, font_scale, text_color, text_thickness)
    
    # Save the image (always save since we can't display)
    if output_path is None:
        # Generate default output path if none provided
        import os
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_dir = os.path.dirname(image_path)
        output_path = os.path.join(output_dir, f"{base_name}_marked.png")
    
    cv2.imwrite(output_path, marked_image)
    print(f"Marked image saved to: {output_path}")
    
    # Print image dimensions and marker info
    height, width = marked_image.shape[:2]
    print(f"Image dimensions: {width}x{height}")
    print(f"Marker placed at coordinates: ({x}, {y})")
    
    return marked_image

# Main execution
if __name__ == "__main__":
    # Image path and coordinates
    image_path = "/home/azureuser/workstation/genfied/ObjectTracking/output_images/store-109/frame_000177_seq173_cv2_standard_20250730_164606_455.png"
    x_coord = 233
    y_coord = 178
    
    # Target dimensions
    target_width = 874
    target_height = 600
    
    print("=== Image Resizing with Same Coordinates ===")
    print(f"Resizing image to {target_width}x{target_height}")
    print(f"Drawing marker at original coordinates: ({x_coord}, {y_coord})")
    
    # Process the image
    result = draw_marker_on_resized_image_no_transform(
        image_path, 
        x_coord, 
        y_coord, 
        target_width, 
        target_height
    )
    
    if result is not None:
        print(f"\nSuccess! Marker drawn at coordinates ({x_coord}, {y_coord}) on {target_width}x{target_height} image")
    else:
        print("Failed to process the image.")



# # Main execution
# if __name__ == "__main__":
#     # Image path and coordinates
#     image_path = "/home/azureuser/workstation/genfied/ObjectTracking/output_images/store-109/frame_000177_seq173_cv2_standard_20250730_164606_455.png"
#     x_coord = 233
#     y_coord = 178
    
#     # Optional: specify output path to save the marked image
#     output_path = "/home/azureuser/workstation/genfied/ObjectTracking/output_images/store-109/frame_000177_seq173_marked.png"
    
#     # Draw marker on the image
#     marked_image = draw_marker_on_image(image_path, x_coord, y_coord, output_path)
    
#     if marked_image is not None:
#         print(f"Successfully marked coordinates ({x_coord}, {y_coord}) on the image!")
#     else:
#         print("Failed to process the image.")