# PDF Government Document Processing

This project demonstrates a custom solution for processing PDF scans of government documents. The application takes a PDF scan as input, detects tables within the document, performs OCR on the tables, and extracts relevant text information. The extracted text is then presented to the user in a format that can be easily copied and pasted into an Excel spreadsheet for further analysis.

## Workflow

The high-level workflow of the PDF government document processing application is as follows:

1. **Input**: The user uploads a PDF file containing scanned images of government documents through the Streamlit web application.

2. **Image Extraction**: The application extracts images from the PDF file and converts them to grayscale numpy arrays.

3. **Preprocessing**: The extracted images undergo various preprocessing steps:
   - Removal of vertical printer lines
   - Auto-deskewing to correct image skew

4. **Table and Cell Detection**: The application detects tables and cells within the preprocessed images and groups the cells by tables.

5. **OCR and Text Extraction**: OCR is performed on the detected cells to extract text.

6. **Text Processing**: The extracted text is adjusted by matching the OCR-generated strings with a predefined dictionary of possible strings using fuzzy search.

7. **Output**: The processed results are displayed in the web app. The user can copy the text into their own table for further use.

## Challenges and Solutions

During the development of this application, several challenges arose due to the nature of scanned government documents. Below are the main issues encountered and how they were addressed:

### 1. Skewed Pages Affecting Table and Cell Detection

In many cases, the scanned documents were slightly tilted, which caused issues in detecting tables and cells accurately. To solve this problem, I developed a function to automatically detect the skew angle of the document and correct it.

- **Approach**: The function applies a Gaussian blur to the image, converts it to a binary format, and detects long, thin contours (which are often aligned with horizontal or vertical lines in the document). By analyzing the angles of these contours, the skew angle is estimated and the image is rotated to correct for the tilt.
  
### 2. Vertical Printer Lines Interfering with OCR

Some documents contained defects in the form of vertical printer lines, which interfered with the OCR process. These lines needed to be detected and removed without affecting the document's content.

- **Approach**: I developed a function to detect and remove vertical lines by applying a combination of adaptive thresholding, morphological operations, and contour filtering. The algorithm focuses on detecting lines at the edges of the page and extrapolates them across the entire document before removing them.

This method uses a stretched morphological kernel to identify vertical lines and filters them based on their height and angle, ensuring that only the unwanted printer lines are removed while preserving the actual content of the document.

### 3. Accurately Determining Table and Cell Coordinates

Detecting the precise coordinates of tables and cells within the document is a non-trivial task. The scanned images often contain noise, varying cell sizes, making it challenging to accurately identify the boundaries of each table and cell.

- **Approach**: I developed a multi-step algorithm that combines various image processing techniques to accurately detect table and cell coordinates. The algorithm enhances the image, detects contours, filters them to identify cells, creates a mask to detect table borders, assigns cells to tables, and refines cell coordinates based on the table structure.

### 4. Merged Cells Due to Poor Scan Quality

In some cases, the poor quality of the scanned documents led to cells being merged during the detection process, resulting in two cells being identified as one.

- **Approach**: Knowing that all tables have a simple structure without merged cells, I utilized the already determined intersection points of the table lines and applied a regression line method to find the intersection points of the lines in the incorrectly identified cells.

## 5. OCR Errors and Typos in Extracted Text

EasyOCR sometimes produced errors and typos in the extracted text, which could lead to inaccuracies in the final output.

- **Approach**: To address this issue, I leveraged the fact that the client has a predefined list of possible strings that can be extracted from the documents. I implemented a function to correct the OCR-generated text by matching it with the closest string from the predefined list using fuzzy string matching techniques.

### 6. Tables Spanning Multiple Pages

In some cases, tables in the scanned documents span across multiple pages, causing the text to be split between two pages. This leads to errors in the digitization process, as the two parts of the split text are processed separately.

- **Approach**: I developed a logic to determine if two tables from adjacent pages are actually one split table and merge them back together. The solution involves sorting tables, determining split tables, marking cells as potentially split, merging text based on fuzzy matching scores, and applying OCR error correction to obtain the final corrected and merged text.