# Standard libraries and file system operations
import os  # Provides a way of using operating system dependent functionality, like reading or writing to the filesystem.

# Image processing
import cv2  # OpenCV, an open-source library for computer vision, machine learning, and image processing.
import numpy as np  # Fundamental package for scientific computing with Python, used here for handling arrays.
from PIL import Image  # Python Imaging Library, adds image processing capabilities to your Python interpreter.
import fitz  # PyMuPDF, a library for accessing PDF files and extracting information like images or text.

# Machine learning
from sklearn.linear_model import LinearRegression  # Provides linear regression model functionality.

# Optical character recognition (OCR)
import pytesseract  # Python-tesseract, an OCR tool for extracting text from images.

# Text processing and error correction
from fuzzywuzzy import fuzz  # Library for fuzzy string matching, useful for text comparison and correction.

# Web application framework
import streamlit as st  # An open-source app framework for Machine Learning and Data Science teams.
import tempfile  # Used for creating temporary files and directories.


def extract_images_from_pdf(pdf_path):
    """Extract images from a PDF file and return them as a list of numpy arrays in grayscale."""
    images = []
    pdf_document = fitz.open(pdf_path)

    for page_index in range(len(pdf_document)):
        page = pdf_document.load_page(page_index)
        images_info = page.get_images(full=True)

        # Check if there is exactly one image on the page
        if len(images_info) != 1:
            pdf_document.close()
            raise ValueError(f"Page {page_index + 1} of the PDF does not contain exactly one image.")

        # Extract the image information
        xref = images_info[0][0]
        base_image = pdf_document.extract_image(xref)
        image_bytes = base_image["image"]

        # Convert the image bytes to a numpy array in grayscale
        np_img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
        images.append(np_img)

    pdf_document.close()
    return images



def remove_vertical_printer_lines(image, block_size=13, c=2, blur_kernel_size=3, crop_params=(50, 12, 35)):
    """Process the entire image to remove vertical printer lines."""
    
    def preprocess_image_portion(cropped_image_portion, blur_kernel_size=3, clahe_clip_limit=2, clahe_tile_grid_size=(1, 35)):
        """Preprocess the cropped portion of the image."""
        cropped_image_portion = cv2.GaussianBlur(cropped_image_portion, (blur_kernel_size, blur_kernel_size), 0)
        clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_tile_grid_size)
        cropped_image_portion = clahe.apply(cropped_image_portion)
        return cropped_image_portion

    def detect_vertical_line_contours(cropped_image_portion, block_size=13, c=2, erosion_kernel_size=(1, 10), dilation_kernel_size=(1, 20)):
        """Detect vertical contours in the cropped portion."""
        crop_bin = cv2.adaptiveThreshold(cropped_image_portion, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, c)
        crop_bin_inv = cv2.bitwise_not(crop_bin)
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, erosion_kernel_size)
        crop_eroded = cv2.erode(crop_bin_inv, vertical_kernel)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, dilation_kernel_size)
        crop_dilate = cv2.dilate(crop_eroded, kernel)
        contours, _ = cv2.findContours(crop_dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def filter_vertical_line_contours(contours, crop_shape, min_height_ratio=0.8, angle_threshold=10):
        """Filter contours based on height and angle."""
        filtered_contours = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if h >= min_height_ratio * crop_shape[0]:
                angle = np.degrees(np.arctan2(h, w))
                if abs(angle - 90) <= angle_threshold:
                    filtered_contours.append(contour)
        return filtered_contours

    def process_image_portion_for_lines(cropped_image_portion, block_size=13, c=2, blur_kernel_size=3, clahe_clip_limit=2,
                     clahe_tile_grid_size=(1, 35), erosion_kernel_size=(1, 10), dilation_kernel_size=(1, 20),
                     min_height_ratio=0.8, angle_threshold=10):
        """Process a cropped portion of the image to detect vertical lines."""
        cropped_image_portion = preprocess_image_portion(cropped_image_portion, blur_kernel_size, clahe_clip_limit, clahe_tile_grid_size)
        contours = detect_vertical_line_contours(cropped_image_portion, block_size, c, erosion_kernel_size, dilation_kernel_size)
        return filter_vertical_line_contours(contours, cropped_image_portion.shape, min_height_ratio, angle_threshold)

    def offset_contour_positions(contours, x_offset, y_offset):
        """Adjust the positions of contours based on offsets."""
        for contour in contours:
            contour[:, :, 0] += x_offset
            contour[:, :, 1] += y_offset
        return contours

    def remove_detected_lines_from_image(image, contours):
        """Remove detected lines from the image."""
        new_image = image.copy()
        height, _ = image.shape

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cx = x + w // 2

            top_point = (cx, 0)
            bottom_point = (cx, height - 1)

            new_contour = np.array([[top_point], [bottom_point]], dtype=np.int32)

            mask = np.zeros_like(image)
            cv2.drawContours(mask, [new_contour], -1, 255, thickness=cv2.FILLED)

            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
            dilated_mask = cv2.dilate(mask, horizontal_kernel, iterations=1)

            for y in range(height):
                if np.any(dilated_mask[y, :] == 255):
                    slice_pixels = image[y, dilated_mask[y, :] == 255]
                    
                    if len(slice_pixels) > 0:
                        median_brightness = np.median(slice_pixels)
                        if median_brightness > 240:
                            new_image[y, dilated_mask[y, :] == 255] = 255

        return new_image

    # Main execution flow
    left_right_offset, top_bottom_offset, rect_height = crop_params

    if blur_kernel_size % 2 == 0:
        blur_kernel_size += 1

    height, width = image.shape

    top_crop = image[top_bottom_offset:top_bottom_offset + rect_height, left_right_offset:width - left_right_offset]
    bottom_crop = image[height - top_bottom_offset - rect_height:height - top_bottom_offset, left_right_offset:width - left_right_offset]

    top_contours = process_image_portion_for_lines(top_crop, block_size, c, blur_kernel_size)
    bottom_contours = process_image_portion_for_lines(bottom_crop, block_size, c, blur_kernel_size)

    top_contours = offset_contour_positions(top_contours, left_right_offset, top_bottom_offset)
    bottom_contours = offset_contour_positions(bottom_contours, left_right_offset, height - top_bottom_offset - rect_height)

    all_contours = top_contours + bottom_contours

    return remove_detected_lines_from_image(image, all_contours)


def correct_image_skew(image):
    """Corrects the skew of the input image."""
    
    # Blur the image
    blurred_image = cv2.GaussianBlur(image, (23, 23), 0)
    
    # Threshold to get a binary image with inverted colors
    binary_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # Dilate the image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
    dilated_image = cv2.dilate(binary_image, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(dilated_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours to find long thin lines
    detected_line_angles = []
    for contour in contours:
        minAreaRect = cv2.minAreaRect(contour)
        (center, (width, height), angle) = minAreaRect
        if width > 100 and height < 20:  # Assuming lines are longer than 100 pixels and thinner than 20 pixels
            if angle < -45:
                angle = 90 + angle
            detected_line_angles.append(angle)
    
    # Calculate the median angle
    if detected_line_angles:
        detected_line_angles.sort()
        mid_angle = detected_line_angles[int(len(detected_line_angles) / 2)]
        
        # Rotate the image
        (h, w) = image.shape[:2]
        center = (w / 2, h / 2)
        m = cv2.getRotationMatrix2D(center, mid_angle, 1)
        deskewed_image = cv2.warpAffine(image, m, (w, h), borderValue=(255, 255, 255))
        return deskewed_image
        
    return image


def enhance_and_clean_faint_scan(image):
    """Enhance and clean up poorly printed faint scans."""

    def binarize_and_create_inverted_mask(image):
        """Binarize the image and create an inverted mask."""
        otsu_thresh, _ = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, binary_image = cv2.threshold(image, otsu_thresh + 40, 255, cv2.THRESH_BINARY)
        inverted_mask = cv2.bitwise_not(binary_image)
        return binary_image, inverted_mask

    def calculate_masked_area_brightness(image, mask_indices):
        """Calculate the average brightness of valid pixels within the mask."""
        valid_mask = (image[mask_indices] >= 0) & (image[mask_indices] <= 245)
        valid_pixel_brightness = image[mask_indices][valid_mask]
        return np.mean(valid_pixel_brightness)

    def apply_contrast_and_gamma_correction(image, mask_indices, gamma=0.8):
        """Apply CLAHE and gamma correction to the image."""
        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(16, 16))
        corrected_image = np.copy(image)
        corrected_image[mask_indices] = clahe.apply(image)[mask_indices]
        
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        corrected_image[mask_indices] = cv2.LUT(corrected_image[mask_indices].reshape(-1, 1), table).reshape(-1)
        corrected_image[corrected_image > 220] = 255
        return corrected_image

    def remove_mask_artifacts(image, inverted_mask):
        """Remove artifacts from the image."""
        dilated_mask = cv2.dilate(inverted_mask, np.ones((7, 7), np.uint8), iterations=1)
        binary_mask = cv2.bitwise_not(dilated_mask)
        image[binary_mask == 255] = 255
        return image

    # Main execution flow
    binary_image, inverted_mask = binarize_and_create_inverted_mask(image)
    mask_indices = np.where(inverted_mask == 255)
    average_brightness = calculate_masked_area_brightness(image, mask_indices)

    if average_brightness > 80:
        corrected_image = apply_contrast_and_gamma_correction(image, mask_indices)
    else:
        corrected_image = np.copy(image)

    corrected_image = remove_mask_artifacts(corrected_image, inverted_mask)

    return corrected_image


def refine_cell_coordinates(cells, table_rect, tolerance=10):
    def group_cells_by_rows_or_columns(cells, tolerance, index):
        """Group cells into rows or columns based on their coordinates."""
        cells = sorted(cells, key=lambda cell: cell[index])
        clusters = []
        current_cluster = []
        current_min = cells[0][index]
    
        for cell in cells:
            if (cell[index] - current_min) <= tolerance:
                current_cluster.append(cell)
            else:
                if current_cluster:
                    clusters.append(current_cluster)
                current_cluster = [cell]
                current_min = cell[index]
        if current_cluster:
            clusters.append(current_cluster)
        return clusters
    
    def compute_regression_lines_for_cells(clusters, range_min, range_max, index):
        """Calculate regression lines for the clusters of cells."""
        lines = []
        for cluster in clusters:
            coords = np.array([cell[index] for cell in cluster]).reshape(-1, 1)
            targets = np.array([cell[1 - index] for cell in cluster])
            if len(cluster) > 1:
                model = LinearRegression()
                model.fit(coords, targets)
                range_vals = np.array([[range_min], [range_max]])
                predictions = model.predict(range_vals)
                if index == 0:
                    lines.append((range_vals.flatten(), predictions))
                else:
                    lines.append((predictions, range_vals.flatten()))
        return lines

    def determine_cell_coordinates_from_lines(horizontal_lines, vertical_lines, table_rect):
        """Determine cell coordinates based on the intersections of regression lines."""
        table_x_min, table_y_min, table_w, table_h = table_rect
        table_x_max = table_x_min + table_w
        table_y_max = table_y_min + table_h

        cell_coordinates = []

        for h_idx, (hx_coords, hy_coords) in enumerate(horizontal_lines):
            for v_idx, (vx_coords, vy_coords) in enumerate(vertical_lines):
                top_left_x = round(vx_coords[0])
                top_left_y = round(hy_coords[0])
                if h_idx < len(horizontal_lines) - 1 and v_idx < len(vertical_lines) - 1:
                    bottom_right_x = round(vertical_lines[v_idx + 1][0][0])
                    bottom_right_y = round(horizontal_lines[h_idx + 1][1][0])
                else:
                    bottom_right_x = round(table_x_max if v_idx == len(vertical_lines) - 1 else vertical_lines[v_idx + 1][0][0])
                    bottom_right_y = round(table_y_max if h_idx == len(horizontal_lines) - 1 else horizontal_lines[h_idx + 1][1][0])

                cell_corners = [
                    (top_left_x, top_left_y),
                    (bottom_right_x, top_left_y),
                    (bottom_right_x, bottom_right_y),
                    (top_left_x, bottom_right_y)
                ]
                cell_coordinates.append({
                    'corners': cell_corners,
                    'row': h_idx,
                    'column': v_idx
                })

        return cell_coordinates

    # Main execution
    # Cluster cells into rows and columns
    rows = group_cells_by_rows_or_columns(cells, tolerance, index=1)
    columns = group_cells_by_rows_or_columns(cells, tolerance, index=0)

    # Define table boundaries
    table_x_min, table_y_min = table_rect[0], table_rect[1]
    table_x_max = table_x_min + table_rect[2]
    table_y_max = table_y_min + table_rect[3]

    # Calculate regression lines
    horizontal_lines = compute_regression_lines_for_cells(rows, table_x_min, table_x_max, index=0)
    vertical_lines = compute_regression_lines_for_cells(columns, table_y_min, table_y_max, index=1)

    # Calculate cell coordinates
    cell_coordinates = determine_cell_coordinates_from_lines(horizontal_lines, vertical_lines, table_rect)

    # Calculate the number of rows and columns and the total number of cells
    num_rows = len(horizontal_lines)
    num_columns = len(vertical_lines)
    total_cells = num_rows * num_columns

    return {
        'cell_coordinates': cell_coordinates,
        'num_rows': num_rows,
        'num_columns': num_columns,
        'total_cells': total_cells
    }


def extract_tables_and_cells(image):
    """Extract tables and their cells from the grayscale image."""
    
    def detect_table_cells(image, min_area=700, max_aspect_ratio=30.0):
        """Detect table cells from the grayscale image."""
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
        gray = clahe.apply(image)
        
        # Apply Gaussian blur to reduce noise before thresholding
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        # Apply adaptive thresholding to get a binary image
        binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours based on area and aspect ratio
        filtered_contours = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area:
                continue

            x, y, w, h = cv2.boundingRect(c)
            aspect_ratio = max(w, h) / min(w, h)
            if aspect_ratio > max_aspect_ratio:
                continue

            filtered_contours.append(c)

        # Find bounding rectangles of the filtered table cells
        cell_bounding_boxes = [cv2.boundingRect(c) for c in filtered_contours]

        # Sort bounding boxes by y coordinate (top to bottom), then by x coordinate (left to right)
        cell_bounding_boxes = sorted(cell_bounding_boxes, key=lambda x: (x[1], x[0]))

        return cell_bounding_boxes

    def detect_table_borders(cell_bounding_boxes, image):
        """Find contours of the outer tables."""
        # Create a mask for the table cells
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        for (x, y, w, h) in cell_bounding_boxes:
            cv2.rectangle(mask, (x, y), (x + w, y + h), 255, thickness=cv2.FILLED)

        # Find contours of the tables
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find bounding rectangles of the tables
        outer_table_bounding_boxes = [cv2.boundingRect(c) for c in contours]

        return outer_table_bounding_boxes

    def assign_cells_to_tables(cell_bounding_boxes, outer_table_bounding_boxes):
        """Assign cells to their respective tables."""
        
        def is_rectangle_inside(inner_box, outer_box):
            """Check if a rectangle is inside another rectangle."""
            ix, iy, iw, ih = inner_box
            ox, oy, ow, oh = outer_box
            return ox <= ix <= ox + ow and oy <= iy <= oy + oh and ox <= ix + iw <= ox + ow and oy <= iy + ih <= oy + oh

        detected_tables = []

        for table_box in outer_table_bounding_boxes:
            tx, ty, tw, th = table_box
            table_cells = []

            for cell_box in cell_bounding_boxes:
                cx, cy, cw, ch = cell_box
                # Check if the cell is within the table
                if tx <= cx and cx + cw <= tx + tw and ty <= cy and cy + ch <= ty + th:
                    table_cells.append(cell_box)

            # Remove cells that contain other cells
            filtered_cells = []
            for cell in table_cells:
                if not any(is_rectangle_inside(inner_cell, cell) for inner_cell in table_cells if inner_cell != cell):
                    filtered_cells.append(cell)

            # Filter out tables with fewer than 8 cells
            if len(filtered_cells) >= 8:
                detected_tables.append({
                    'table_box': table_box,
                    'cells': filtered_cells
                })

        return detected_tables

    # Main execution
    cell_bounding_boxes = detect_table_cells(image)
    outer_table_bounding_boxes = detect_table_borders(cell_bounding_boxes, image)
    detected_tables = assign_cells_to_tables(cell_bounding_boxes, outer_table_bounding_boxes)

    refined_tables = []
    for table in detected_tables:
        table_rect = table['table_box']
        cells = table['cells']
        refined_data = refine_cell_coordinates(cells, table_rect)
        refined_tables.append({
            'table_box': table_rect,
            'cells': refined_data['cell_coordinates'],
            'num_rows': refined_data['num_rows'],
            'num_columns': refined_data['num_columns'],
            'total_cells': refined_data['total_cells']
        })

    return refined_tables


def extract_text_from_table_cells(image, table_info):
    """Extract text from the second column of table cells in the image."""

    def extract_raw_text_from_cells(image, cells):
        """Extract raw text from the specified cells without correction."""
        digitized_data = []
        for cell in cells:
            if cell['column'] == 1:  # Second column (0-indexed)
                # Extract the region of interest (ROI) for each cell using cell coordinates
                x1, y1 = cell['corners'][0]
                x2, y2 = cell['corners'][2]
                roi = image[y1:y2, x1:x2]  # Slice the numpy array to get the ROI

                # Use Tesseract to extract text from the cell ROI
                text = pytesseract.image_to_string(roi, lang='rus', config="--psm 6")

                # Replace newlines with spaces
                cleaned_text = text.replace('\n', ' ').replace('\r', ' ').strip()

                digitized_data.append({
                    'row': cell['row'],
                    'column': cell['column'],
                    'text': cleaned_text  # Store raw text without correction
                })

        return digitized_data

    def is_suspected_split(image, table, table_index, total_tables):
        """Check if the current table is suspected to be a continuation from a previous page or to continue on the next page."""
        suspected_split = False
        height, width = image.shape

        def check_text_presence(crop):
            """Helper function to check if text is present in the cropped image portion."""
            text = pytesseract.image_to_string(crop, lang='rus', config="--psm 6")
            return len(text.strip()) < 10

        # Checking if the table is the first on the page
        if table_index == 0:
            # Coordinates for the crop region above the table
            first_cell = table['cells'][0]
            top_left_y = first_cell['corners'][0][1]
            top_crop_y = max(0, top_left_y - 300)
            top_crop = image[top_crop_y:top_left_y, 0:width]

            # Check for absence of text above the first table
            if check_text_presence(top_crop):
                suspected_split = True

        # Checking if the table is the last on the page
        if table_index == total_tables - 1:
            # Coordinates for the crop region below the table
            last_cell = table['cells'][-1]
            bottom_left_y = last_cell['corners'][2][1]
            bottom_crop_y = min(height, bottom_left_y + 300)
            bottom_crop = image[bottom_left_y:bottom_crop_y, 0:width]

            # Check for absence of text below the last table
            if check_text_presence(bottom_crop):
                suspected_split = True

        return suspected_split

    all_tables_data = []

    # Sort tables by their top-left Y coordinate (table_box[1]) in ascending order
    table_info.sort(key=lambda x: x['table_box'][1])

    for i, table in enumerate(table_info):
        if table['num_columns'] == 6:
            # Digitize the content of each cell from the second column
            digitized_data = extract_raw_text_from_cells(image, table['cells'])

            # Check if the table is suspected to be split
            suspected_split = is_suspected_split(image, table, i, len(table_info))

            table_data = {
                'table_num': i + 1,
                'cells': digitized_data,
                'suspected_split': suspected_split
            }
            all_tables_data.append(table_data)

    return all_tables_data


def remove_similar_phrases(text, reference_phrase):
    """Remove phrases similar to the reference phrase from the text."""
    words = text.split()
    clean_words = []
    for word in words:
        score = fuzz.partial_ratio(word, reference_phrase)
        if score < 70:
            clean_words.append(word)
    return ' '.join(clean_words)

def correct_ocr_errors(text, correction_dictionary):
    """Correct OCR errors in the text using a correction dictionary."""
    closest_match = None
    best_score = 0
    for correct_string in correction_dictionary:
        score = fuzz.ratio(text, correct_string)
        if score > best_score:
            closest_match = correct_string
            best_score = score

    # If score is less than 30, return an empty string
    if best_score < 30:
        closest_match = ""

    return closest_match, best_score


def merge_tables(all_tables_data):
    # Load service descriptions
    with open('service_descriptions.txt', 'r', encoding='utf-8') as f:
        correction_dictionary = [line.strip() for line in f]
    
    # Standard phrase to remove
    reference_phrase = "(полустационарное обслуживание)"

    merged_table = []
    previous_table_was_split = False

    for page_index, page_tables in enumerate(all_tables_data):
        for i, table in enumerate(page_tables):
            if not table['suspected_split']:
                # Add all rows from a non-split table directly
                merged_table.extend([cell['text'] for cell in table['cells'] if cell['text']])
                previous_table_was_split = False  # Reset the flag
            else:
                if previous_table_was_split:
                    # Handle merging of split tables
                    prev_last_row = merged_table[-1]
                    current_second_row = table['cells'][1]['text']

                    # Perform fuzzy matching using correct_ocr_errors
                    best_str1, score1 = correct_ocr_errors(prev_last_row, correction_dictionary)
                    best_str2, score2 = correct_ocr_errors(current_second_row, correction_dictionary)

                    combined_row = ' '.join([prev_last_row, current_second_row])
                    best_str3, score3 = correct_ocr_errors(combined_row, correction_dictionary)

                    # Decide how to merge based on the rules
                    if best_str1 and best_str2 and best_str1 != best_str2:
                        merged_table[-1] = prev_last_row
                        merged_table.append(current_second_row)
                    elif not best_str1 and not best_str2:
                        if best_str3:
                            merged_table[-1] = combined_row
                    elif best_str1 == best_str2:
                        if score3 > score1 and score3 > score2:
                            merged_table[-1] = combined_row
                        else:
                            merged_table[-1] = prev_last_row

                    # Add remaining rows from the current table except the first row
                    merged_table.extend([cell['text'] for cell in table['cells'][2:] if cell['text']])
                    previous_table_was_split = False
                else:
                    # Check if the current table is the first in the document and suspected to be split
                    if page_index == 0 and i == 0:
                        merged_table.extend([cell['text'] for cell in table['cells'] if cell['text']])
                        previous_table_was_split = True
                    else:
                        # Starting a new split table, add the first row
                        merged_table.extend([cell['text'] for cell in table['cells'] if cell['text']])
                        previous_table_was_split = True

    # Correct the merged text data
    corrected_table = []
    #print(merged_table)
    for row in merged_table:
        # Remove similar phrases
        cleaned_row = remove_similar_phrases(row, reference_phrase)
        # Correct OCR errors
        corrected_row, _ = correct_ocr_errors(cleaned_row, correction_dictionary)
        # Check for rows similar to "Наименование услуги"
        _, name_score = correct_ocr_errors(corrected_row, ["Наименование услуги"])
        #print(f"row: {row}\ncleaned_row {cleaned_row}\ncorrected_row {corrected_row}\nname_score {name_score}\n\n")
        if name_score > 70:
            continue  # Skip this row only if the similarity score is above 85
        # Add to the corrected table
        corrected_table.append(corrected_row)

    return corrected_table


def pdf_text_extraction_workflow(pdf_path):
    """Process a PDF file to extract text from tables, handling scanned images and various preprocessing steps."""
    
    # Initialize the array to hold text from all tables.
    all_texts = []

    # Extract images from the PDF.
    images = extract_images_from_pdf(pdf_path)

    # Loop through each image (page) in the PDF.
    for page_number, image in enumerate(images):
        # Step 1: Remove vertical printer lines.
        image_without_lines = remove_vertical_printer_lines(image)
        
        # Step 2: Correct image skew.
        deskewed_image = correct_image_skew(image_without_lines)
        
        # Step 3: Enhance and clean faint scans.
        enhanced_image = enhance_and_clean_faint_scan(deskewed_image)
        
        # Step 4: Extract tables and their cells.
        table_info = extract_tables_and_cells(enhanced_image)
        
        # Step 4: Extract text from the table cells.
        extracted_text = extract_text_from_table_cells(enhanced_image, table_info)

        # Step 5: Add the extracted text to the all_texts list.
        all_texts.append(extracted_text if extracted_text else [])

    # Step 6: Merge texts from all tables across all pages.    
    merged_tables_text = merge_tables(all_texts)
    
    return merged_tables_text

def main():
    st.title("PDF Table Digitization App")

    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    if uploaded_file is not None:
        if st.button("Process PDF"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
                temp_pdf.write(uploaded_file.read())
                temp_pdf_path = temp_pdf.name
            
            # Run the workflow function with the temporary file path
            results = pdf_text_extraction_workflow(temp_pdf_path)
            
            # Display the results
            st.text_area("Extracted Text", value='\n'.join(results), height=300)
            
            # Optionally, clean up the temporary file
            os.remove(temp_pdf_path)

if __name__ == "__main__":
    main()