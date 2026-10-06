import cv2
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

# Project folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Output folder
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

image = None
image_path = None


# 1. Read / Select Image
def read_image():
    global image, image_path

    root = tk.Tk()
    root.withdraw()

    image_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[
            ("Image Files", "*.jpg *.jpeg *.png *.bmp"),
            ("All Files", "*.*")
        ]
    )

    root.destroy()

    if not image_path:
        print("No image selected.")
        return

    # Read image safely
    image_data = np.fromfile(image_path, dtype=np.uint8)
    image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

    if image is None:
        print("Error: Could not read image.")
        return

    print("\nImage loaded successfully!")
    print("Selected:", os.path.basename(image_path))


# 2. Display Image Properties
def display_properties():
    if image is None:
        print("Please select an image first.")
        return

    height, width = image.shape[:2]
    channels = image.shape[2] if len(image.shape) == 3 else 1

    print("\n================================")
    print("       IMAGE PROPERTIES")
    print("================================")
    print("Width     :", width)
    print("Height    :", height)
    print("Channels  :", channels)
    print("Data Type :", image.dtype)


# 3. Display Image
def display_image():
    if image is None:
        print("Please select an image first.")
        return

    cv2.imshow("Original Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 4. Crop Image
def crop_image():
    if image is None:
        print("Please select an image first.")
        return

    height, width = image.shape[:2]

    x1 = width // 4
    y1 = height // 4
    x2 = 3 * width // 4
    y2 = 3 * height // 4

    cropped = image[y1:y2, x1:x2]

    cv2.imwrite(
        os.path.join(OUTPUT_DIR, "cropped.jpg"),
        cropped
    )

    print("Cropped image saved.")

    cv2.imshow("Cropped Image", cropped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 5. Resize Image
def resize_image():
    if image is None:
        print("Please select an image first.")
        return

    resized = cv2.resize(image, (500, 500))

    cv2.imwrite(
        os.path.join(OUTPUT_DIR, "resized.jpg"),
        resized
    )

    print("Resized image saved.")

    cv2.imshow("Resized Image", resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 6. Convert to Grayscale
def grayscale_image():
    if image is None:
        print("Please select an image first.")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cv2.imwrite(
        os.path.join(OUTPUT_DIR, "grayscale.jpg"),
        gray
    )

    print("Grayscale image saved.")

    cv2.imshow("Grayscale Image", gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 7. Save Image
def save_image():
    if image is None:
        print("Please select an image first.")
        return

    cv2.imwrite(
        os.path.join(OUTPUT_DIR, "processed_image.jpg"),
        image
    )

    print("Processed image saved.")


# 8. Store Image Details in JSON
def save_json():
    if image is None:
        print("Please select an image first.")
        return

    height, width = image.shape[:2]
    channels = image.shape[2] if len(image.shape) == 3 else 1

    data = {
        "filename": os.path.basename(image_path),
        "width": width,
        "height": height,
        "channels": channels,
        "data_type": str(image.dtype)
    }

    json_path = os.path.join(BASE_DIR, "image_data.json")

    with open(json_path, "w") as file:
        json.dump(data, file, indent=4)

    print("Image information saved to image_data.json")


# 9. Display Data using Pandas
def view_data():
    json_path = os.path.join(BASE_DIR, "image_data.json")

    try:
        with open(json_path, "r") as file:
            data = json.load(file)

        df = pd.DataFrame([data])

        print("\n================================")
        print("          IMAGE DATA")
        print("================================")

        print(df.to_string(index=False))

    except FileNotFoundError:
        print("Please select option 8 first.")


# 10. Generate Histogram
def generate_graph():
    if image is None:
        print("Please select an image first.")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    plt.figure(figsize=(8, 5))

    plt.hist(
        gray.ravel(),
        bins=256,
        range=[0, 256],
        color="blue"
    )

    plt.title("Image Intensity Histogram")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")

    graph_path = os.path.join(
        OUTPUT_DIR,
        "image_histogram.png"
    )

    plt.savefig(graph_path)

    print("Graph saved.")

    plt.show()


# Main Menu
def menu():

    while True:

        print("\n================================")
        print("        IMAGE ANALYZER")
        print("================================")

        print("1. Read Image")
        print("2. Display Image Properties")
        print("3. Display Image")
        print("4. Crop Image")
        print("5. Resize Image")
        print("6. Convert to Grayscale")
        print("7. Save Image")
        print("8. Store Image Details in JSON")
        print("9. View Image Data using Pandas")
        print("10. Generate Graph")
        print("11. Exit")

        choice = input("\nEnter your choice: ")

        if choice == "1":
            read_image()

        elif choice == "2":
            display_properties()

        elif choice == "3":
            display_image()

        elif choice == "4":
            crop_image()

        elif choice == "5":
            resize_image()

        elif choice == "6":
            grayscale_image()

        elif choice == "7":
            save_image()

        elif choice == "8":
            save_json()

        elif choice == "9":
            view_data()

        elif choice == "10":
            generate_graph()

        elif choice == "11":
            print("Thank you!")
            break

        else:
            print("Invalid choice.")


# Start program
menu()