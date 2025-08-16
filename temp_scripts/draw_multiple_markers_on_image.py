import cv2

# Path to your input image
input_path = '/home/azureuser/workstation/genfied/ObjectTracking/output_images/test_images/camera_141_20250724121606.jpg'
# Path where the output will be saved
output_path = '/home/azureuser/workstation/genfied/ObjectTracking/output_images/test_images/camera_141_20250724121606_with_points.jpg'

# List of your points (ignoring n and w)
points_data = [
    {"x": 1160.0, "y": 821.0}, {"x": 1159.0, "y": 825.0},
    {"x": 1180.0, "y": 799.0}, {"x": 1181.0, "y": 803.0},
    {"x": 1172.0, "y": 805.0}, {"x": 1201.0, "y": 796.0},
    {"x": 1181.0, "y": 810.0}, {"x": 1189.0, "y": 838.0},
    {"x": 1201.0, "y": 795.0}, {"x": 1166.0, "y": 784.0},
    {"x": 1145.0, "y": 755.0}, {"x": 1148.0, "y": 797.0},
    {"x": 1144.0, "y": 615.0}, {"x": 1281.0, "y": 1037.0},
    {"x": 1307.0, "y": 1058.0}, {"x": 1296.0, "y": 1061.0},
    {"x": 1291.0, "y": 1064.0}, {"x": 1320.0, "y": 898.0},
    {"x": 1260.0, "y": 1042.0}, {"x": 1267.0, "y": 1056.0},
    {"x": 1263.0, "y": 912.0}, {"x": 1257.0, "y": 880.0},
    {"x": 1253.0, "y": 920.0}, {"x": 1250.0, "y": 1001.0},
    {"x": 1254.0, "y": 1028.0}, {"x": 1261.0, "y": 1043.0},
    {"x": 1257.0, "y": 1037.0}, {"x": 1264.0, "y": 910.0},
    {"x": 1325.0, "y": 1004.0}, {"x": 1339.0, "y": 977.0},
    {"x": 1301.0, "y": 952.0}, {"x": 1290.0, "y": 951.0},
    {"x": 1292.0, "y": 948.0}, {"x": 1359.0, "y": 984.0},
    {"x": 1359.0, "y": 985.0}, {"x": 1312.0, "y": 934.0},
    {"x": 1280.0, "y": 1028.0}, {"x": 1192.0, "y": 997.0},
    {"x": 1202.0, "y": 1003.0}, {"x": 1210.0, "y": 1000.0},
    {"x": 1236.0, "y": 1026.0}, {"x": 1269.0, "y": 1036.0},
    {"x": 1271.0, "y": 1034.0}, {"x": 1271.0, "y": 1035.0},
    {"x": 577.0,  "y": 717.0}, {"x": 571.0,  "y": 679.0},
    {"x": 572.0,  "y": 686.0}, {"x": 571.0,  "y": 680.0},
    {"x": 1144.0, "y": 616.0}
]

# Convert to integer tuples
points = [(int(p['x']), int(p['y'])) for p in points_data]

# Load image
img = cv2.imread(input_path)
if img is None:
    raise FileNotFoundError(f"Could not load image at {input_path}")

# Draw each point
for (x, y) in points:
    # radius=5 px, color=red (BGR), filled circle (thickness=-1)
    cv2.circle(img, (x, y), radius=5, color=(0, 0, 255), thickness=-1)

# Save the annotated image
success = cv2.imwrite(output_path, img)
if not success:
    raise IOError(f"Could not write image to {output_path}")
print(f"Annotated image saved to: {output_path}")
