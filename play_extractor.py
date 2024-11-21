import cv2
import numpy as np
import os
import pytesseract
from PIL import Image
import re
import shutil


def process_all_images(input_folder, output_folder, parsed_folder, color_tolerance=50):
    if not os.path.exists(input_folder):
        print(f"Error: The input folder '{input_folder}' does not exist.")
        return

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if not os.path.exists(parsed_folder):
        os.makedirs(parsed_folder)

    for filename in os.listdir(input_folder):
        image_path = os.path.join(input_folder, filename)

        if not filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
            print(f"Skipping non-image file: {filename}")
            continue

        first_sub_image_text = extract_boxes_from_image(
            image_path, output_folder, color_tolerance
        )

        if first_sub_image_text:
            parsed_image_name = sanitize_filename(first_sub_image_text)
        else:
            print(
                f"Could not determine a name for the parsed image. Skipping {filename}."
            )
            continue

        base_name, ext = os.path.splitext(filename)
        parsed_path = os.path.join(parsed_folder, f"{parsed_image_name}{ext}")

        counter = 1
        while os.path.exists(parsed_path):
            parsed_path = os.path.join(
                parsed_folder, f"{parsed_image_name}_{counter}{ext}"
            )
            counter += 1

        shutil.move(image_path, parsed_path)


def extract_boxes_from_image(image_path, output_folder, color_tolerance=50):
    image = cv2.imread(image_path)

    if image is None:
        print(
            f"Error: Unable to load image at '{image_path}'. Please check the file path and try again."
        )
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=11,
        C=2,
    )

    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    box_count = 0
    plays_found = []
    first_sub_image_text = None

    image_replaced = False

    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)

            if w > 30 and h > 30:
                box_image = image[y : y + h, x : x + w]

                trimmed_box = trim_edges_by_color_tolerance(box_image, color_tolerance)

                if trimmed_box.size == 0:
                    print(
                        f"Warning: Trimmed image is empty for box {box_count}, skipping save."
                    )
                    continue

                extracted_text = extract_text_from_image(trimmed_box)

                processed_text = process_extracted_text(extracted_text)

                lines = processed_text.split("\n")
                if len(lines) > 0 and not first_sub_image_text:
                    first_sub_image_text = lines[0]

                if len(lines) < 2:
                    print(
                        f"Not enough text lines to create folder and filename for box {box_count}, skipping."
                    )
                    print(lines)
                    continue

                folder_name = sanitize_filename(lines[0])
                file_name = sanitize_filename(lines[1])

                final_output_folder = os.path.join(output_folder, folder_name)
                if not os.path.exists(final_output_folder):
                    os.makedirs(final_output_folder)

                output_path = os.path.join(final_output_folder, f"{file_name}.png")

                if os.path.exists(output_path):
                    print(f"Replacing existing image: {output_path}")
                    image_replaced = True

                cv2.imwrite(output_path, trimmed_box)
                print(f"Saved {output_path}")

                box_count += 1
                plays_found.append(f"{lines[0]} - {lines[1]}")

    if box_count < 3:
        print(f"Image '{image_path}' has fewer than 3 boxes. Boxes found: {box_count}")

    return first_sub_image_text


def trim_edges_by_color_tolerance(image, color_tolerance):
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    L_channel = lab_image[:, :, 0]

    light_threshold = 255 - color_tolerance

    top = 0
    for i in range(L_channel.shape[0]):
        if np.any(L_channel[i, :] > light_threshold):
            top = i
            break

    bottom = L_channel.shape[0]
    for i in range(L_channel.shape[0] - 1, -1, -1):
        if np.any(L_channel[i, :] > light_threshold):
            bottom = i
            break

    left = 0
    for j in range(L_channel.shape[1]):
        if np.any(L_channel[:, j] > light_threshold):
            left = j
            break

    right = L_channel.shape[1]
    for j in range(L_channel.shape[1] - 1, -1, -1):
        if np.any(L_channel[:, j] > light_threshold):
            right = j
            break

    if top >= bottom or left >= right:
        print("Warning: Trimming resulted in invalid bounds; returning empty image.")
        return np.array([])

    trimmed_image = image[top:bottom, left:right]
    return trimmed_image


def extract_text_from_image(image):
    x_start, y_start = 135, 0
    y_end = 150
    x_end = image.shape[1]

    if x_start >= x_end or y_start >= y_end:
        print("Invalid region for text extraction; skipping.")
        return ""

    roi = image[y_start:y_end, x_start:x_end]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    pil_image = Image.fromarray(thresh)
    text = pytesseract.image_to_string(pil_image)

    return text


def process_extracted_text(text):
    text = text.lower()
    text = text.replace(" ", "_")
    lines = text.splitlines()
    non_empty_lines = [line for line in lines if line.strip() != ""]
    processed_text = "\n".join(non_empty_lines)
    return processed_text


def sanitize_filename(name):
    sanitized_name = re.sub(r"[^\w\-]", "_", name)
    return sanitized_name


process_all_images(
    "./play-images/images-to-parse",
    "./play-images/formations",
    "./play-images/parsed-images",
    color_tolerance=200,
)
