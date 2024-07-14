import os
import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE
from skimage.feature import local_binary_pattern
import joblib
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

# Function to extract features using HOG and LBP
def extract_features(img):
    img_resized = cv2.resize(img, (128, 128))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    hog = cv2.HOGDescriptor()
    hog_features = hog.compute(gray).flatten()
    
    radius = 1
    num_points = 8 * radius
    lbp = local_binary_pattern(gray, num_points, radius, method='uniform')
    lbp_hist, _ = np.histogram(lbp, bins=np.arange(0, num_points + 3), range=(0, num_points + 2))
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= (lbp_hist.sum() + 1e-6)

    features = np.hstack([hog_features, lbp_hist])
    return features

# Data augmentation function
def augment_image(img):
    augmented_images = [img]
    augmented_images.append(cv2.flip(img, 1))  # Horizontal flip
    augmented_images.append(cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE))  # 90-degree rotation
    augmented_images.append(cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE))  # 90-degree counter rotation
    augmented_images.append(cv2.GaussianBlur(img, (5, 5), 0))  # Gaussian blur
    return augmented_images

# Function to correct pixelation using advanced methods
def correct_pixelation(image, scale_factor=2):
    # Apply bilateral filter to reduce noise and keep edges
    filtered_image = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
    
    # Resize the image using cubic interpolation
    height, width = image.shape[:2]
    new_height, new_width = height * scale_factor, width * scale_factor
    corrected_image = cv2.resize(filtered_image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
    return corrected_image

# Function to generate pixelated image for testing
def generate_pixelated_image(image, pixelation_level):
    height, width = image.shape[:2]
    temp_height, temp_width = height // pixelation_level, width // pixelation_level
    temp_image = cv2.resize(image, (temp_width, temp_height), interpolation=cv2.INTER_LINEAR)
    pixelated_image = cv2.resize(temp_image, (width, height), interpolation=cv2.INTER_NEAREST)
    return pixelated_image

# Function to evaluate pixelation correction using super-resolution techniques
def evaluate_pixelation_correction(test_image_path, pixelation_levels):
    test_image = cv2.imread(test_image_path)
    if test_image is None:
        raise ValueError(f"Image '{test_image_path}' could not be loaded.")
    
    for level in pixelation_levels:
        pixelated_image = generate_pixelated_image(test_image, level)

        # Apply bilateral filter for noise reduction and edge preservation
        filtered_image = cv2.bilateralFilter(pixelated_image, d=9, sigmaColor=300, sigmaSpace=75)
        
        # Perform basic interpolation-based super-resolution
        height, width = filtered_image.shape[:2]
        new_height, new_width = height * 2, width * 2
        corrected_image = cv2.resize(filtered_image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        plt.imshow(cv2.cvtColor(corrected_image, cv2.COLOR_BGR2RGB))
        plt.title(f"Corrected Image - Level {level}")
        plt.axis('off')
        plt.show()

# Paths to directories
non_pixelated_dir = r'C:\Users\DELL\intel\ORIGINAL_final'
pixelated_dir = r'C:\Users\DELL\intel\PIXELATED_final'
test_image_path = r'C:\Users\DELL\intel\ORIGINAL_final\Human (5).jpg'

# Check if directories exist
if not os.path.isdir(non_pixelated_dir):
    raise ValueError(f"Directory '{non_pixelated_dir}' does not exist.")
if not os.path.isdir(pixelated_dir):
    raise ValueError(f"Directory '{pixelated_dir}' does not exist.")

# Decimate images by a skip factor (2) for two different offsets (0 and 1)
test_image = cv2.imread(test_image_path)
dec2A = test_image[::2, ::2]
dec2B = test_image[1::2, 1::2]

# Get mean of absolute difference
diff2 = cv2.absdiff(dec2A, dec2B)
mean2 = np.mean(diff2)

print('Mean absdiff image:', mean2)

threshold = 5.0  # This value may need to be adjusted based on your specific use case

if mean2 > threshold:
    # Load images, create labels and apply augmentation
    images = []
    labels = []

    for filename in os.listdir(non_pixelated_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img = cv2.imread(os.path.join(non_pixelated_dir, filename))
            if img is not None:
                for aug_img in augment_image(img):
                    images.append(aug_img)
                    labels.append(0)

    for filename in os.listdir(pixelated_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img = cv2.imread(os.path.join(pixelated_dir, filename))
            if img is not None:
                for aug_img in augment_image(img):
                    images.append(aug_img)
                    labels.append(1)

    images, labels = shuffle(images, labels, random_state=42)

    # Extract features for all images
    X = [extract_features(img) for img in images]
    X = np.array(X)
    y = np.array(labels)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Handle class imbalance using SMOTE
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

    # Train a logistic regression model with regularization and cross-validation
    model = LogisticRegression(max_iter=200, C=1.0, penalty='l2', solver='liblinear')  # Added regularization
    cross_val_scores = cross_val_score(model, X_train_res, y_train_res, cv=5)

    print(f'Cross-Validation Scores: {cross_val_scores}')
    print(f'Average Cross-Validation Score: {cross_val_scores.mean():.4f}')

    model.fit(X_train_res, y_train_res)

    # Save the model and scaler
    joblib.dump(model, 'pixelation_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')

    # Predict and evaluate
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print(f'F1 Score: {f1:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')

    # Evaluate pixelation correction on the test image
    pixelation_levels = [2]  # Define various levels of pixelation
    evaluate_pixelation_correction(test_image_path, pixelation_levels)

else:
    print("The image is not pixelated.")