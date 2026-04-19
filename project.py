# ================================ 
# Step 1: Import Libraries 
# ================================ 
import numpy as np 
import pandas as pd 
import cv2 
import os 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split 
from sklearn.utils import shuffle 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization 
import gc
# ================================ 
# Step 2: Load Dataset 
# ================================ 
df = pd.read_csv(r"C:\Users\vmanv\Desktop\Project\Skin Cancer Detection.csv")  
print(df.head()) 
print("Columns:", df.columns) 

# ================================ 
# Step 3: Image Folder Path 
# ================================ 
image_folder = r"C:\Users\vmanv\Desktop\Project\images"  
print("Folder exists:", os.path.exists(image_folder)) 
print("Sample files:", os.listdir(image_folder)[:5]) 

# ================================ 
# Step 4: Detect Image Column 
# ================================ 
image_col = None 
for col in df.columns: 
    if 'image' in col.lower() or 'file' in col.lower(): 
        image_col = col 
        break 
if image_col is None: 
    raise ValueError("No image column found in CSV") 
print("Using image column:", image_col) 

# ================================ 
# Step 5: Label Column Check 
# ================================ 
label_col = 'BCC'
if label_col not in df.columns:
    raise ValueError(f"Column '{label_col}' not found in CSV")

# ================================ 
# Step 6: Hair Removal Function 
# ================================ 
def remove_hair(image): 
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(9,9)) 
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel) 
    _, thresh = cv2.threshold(blackhat,10,255,cv2.THRESH_BINARY) 
    result = cv2.inpaint(image, thresh, 1, cv2.INPAINT_TELEA) 
    return result 

# ================================ 
# Step 7: Load Images + Labels 
# ================================ 
images = [] 
labels = [] 
for root, dirs, files in os.walk(image_folder): 
    for img_name in files: 

        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')): 
            continue 

        img_path = os.path.join(root, img_name) 

        image = cv2.imread(img_path) 
        if image is None: 
            print("Failed to read:", img_path) 
            continue
        image = cv2.resize(image, (128,128)) 

        # Match with CSV 
        img_id = os.path.splitext(img_name)[0] 
        row = df[df[image_col] == img_id] 
        if len(row) == 0: 
            row = df[df[image_col] == img_name] 
        if len(row) > 0: 
            label = row[label_col].values[0] 
            images.append(image.astype(np.uint8))   
            labels.append(label)
        else: 
            print("No label for:", img_name)

# Convert to numpy 
images = np.array(images, dtype=np.uint8)   
labels = np.array(labels).astype(int) 

# ================================ 
# Step 8: Safety Check 
# ================================ 
print("Total Images Loaded:", len(images)) 
print("Total Labels Loaded:", len(labels)) 

if len(images) == 0: 
    raise ValueError("No images loaded. Check folder path.")

if len(labels) == 0: 
    raise ValueError("No labels matched. Check CSV mapping.")

# ================================ 
# Step 9: Preprocessing 
# ================================ 
processed_images = []
for img in images:
    img = remove_hair(img)
    img = img.astype(np.float32) / 255.0   
    processed_images.append(img)
images = np.array(processed_images, dtype=np.float32)
# Free memory
del processed_images
gc.collect()

# Shuffle data 
images, labels = shuffle(images, labels, random_state=42) 

print("Preprocessing Done") 
print("Images shape:", images.shape) 

# Check class distribution
print("Label distribution:", np.unique(labels, return_counts=True))

# ================================ 
# Step 10: Train-Test Split 
# ================================ 
X_train, X_test, y_train, y_test = train_test_split( 
    images, labels, test_size=0.2, random_state=42 
) 

# ================================ 
# Step 11: Build CNN Model 
# ================================ 
model = Sequential() 
model.add(Conv2D(32,(3,3),activation='relu',input_shape=(128,128,3)))  
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2)) 
model.add(Conv2D(64,(3,3),activation='relu')) 
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2)) 
model.add(Conv2D(128,(3,3),activation='relu')) 
model.add(MaxPooling2D(2,2)) 
model.add(Flatten()) 
model.add(Dense(128,activation='relu')) 
model.add(Dropout(0.5)) 
model.add(Dense(1,activation='sigmoid')) 

# ================================ 
# Step 12: Compile Model 
# ================================ 
model.compile( 
    optimizer='adam', 
    loss='binary_crossentropy', 
    metrics=['accuracy'] 
) 
model.summary() 

# ================================ 
# Step 13: Train Model 
# ================================ 
history = model.fit( 
    X_train, y_train, 
    validation_data=(X_test, y_test), 
    epochs=10, 
    batch_size=16
) 

# ================================ 
# Step 14: Evaluate Model 
# ================================ 
loss, accuracy = model.evaluate(X_test, y_test) 
print("Test Accuracy:", accuracy) 

# ================================ 
# Step 15: Prediction 
# ================================ 
prediction = model.predict(X_test[0:1]) 

# ================================ 
# Step 16: Save Model 
# ================================ 
model.save("skin_cancer_model.keras") 
print("Model saved successfully!")

# ================================ 
# Step 17: Show Result 
# ================================
plt.imshow(X_test[0]) 
plt.title(f"Prediction: {prediction[0][0]:.2f}") 
plt.axis("off") 
plt.show() 

# ================================ 
# Step 18: Plot Accuracy & Loss 
# ================================ 
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.title("Accuracy Graph")
plt.show()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title("Loss Graph")
plt.show()

