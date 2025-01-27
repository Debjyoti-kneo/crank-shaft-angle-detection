import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from io import BytesIO

st.title("Blade Angle Detection App")
st.write("Upload an image of a crankshaft fan to detect blades and compute angles.")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Adaptive Thresholding
    adaptive_thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Find contours and create a filled mask
    contours, _ = cv2.findContours(adaptive_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(gray)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
    
    # Extract pixel coordinates from the segmented fan region
    y_coords, x_coords = np.where(mask == 255)
    pixels = np.column_stack((x_coords, y_coords))
    
    # Apply K-Means Clustering to segment into 3 blade clusters
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(pixels)
    labels = kmeans.labels_
    
    # Detect the fan's center using moments
    M = cv2.moments(largest_contour)
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    
    # Fit a line through each cluster and find blade edges
    lines = []
    for i in range(3):
        cluster_points = pixels[labels == i]
        
        # Fit line to the cluster
        [vx, vy, x0, y0] = cv2.fitLine(cluster_points, cv2.DIST_L2, 0, 0.01, 0.01)
        
        # Find the farthest point from the center within this cluster
        distances = np.sqrt((cluster_points[:, 0] - cx) ** 2 + (cluster_points[:, 1] - cy) ** 2)
        farthest_idx = np.argmax(distances)
        edge_x, edge_y = cluster_points[farthest_idx]
        
        # Store the line from the center to the blade edge
        lines.append((cx, cy, edge_x, edge_y))
    
    # Draw lines from center to blade edges
    output = image.copy()
    cv2.circle(output, (cx, cy), 5, (0, 0, 255), -1)
    for x1, y1, x2, y2 in lines:
        cv2.line(output, (x1, y1), (x2, y2), (255, 0, 0), 2)
    
    # Compute angles between the three lines
    def compute_angle(line1, line2):
        vec1 = np.array([line1[2] - line1[0], line1[3] - line1[1]])
        vec2 = np.array([line2[2] - line2[0], line2[3] - line2[1]])
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        cos_theta = dot_product / (norm1 * norm2)
        angle = np.arccos(np.clip(cos_theta, -1.0, 1.0)) * (180 / np.pi)
        return angle
    
    angle1 = compute_angle(lines[0], lines[1])
    angle2 = compute_angle(lines[1], lines[2])
    angle3 = compute_angle(lines[2], lines[0])
    angles = [angle1, angle2, angle3]
    
    # Add angle labels at midpoints
    for i in range(3):
        line1, line2 = lines[i], lines[(i+1) % 3]
        mx = int((line1[2] + line2[2]) / 2)
        my = int((line1[3] + line2[3]) / 2)
        cv2.putText(output, f"{angles[i]:.1f}", (mx, my), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # Display processed image with detected angles
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    ax.set_title(f'Final Blade Lines (Angles in degrees)')
    ax.axis('off')
    
    st.pyplot(fig)
