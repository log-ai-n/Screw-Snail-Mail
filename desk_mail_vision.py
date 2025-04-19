#!/usr/bin/env python3
"""
Show‑and‑Tell Mail – Vision Edition

This version directly utilizes Gemini's vision capabilities to analyze mail documents
without relying on OCR first. It includes:

1. Captures mail directly through your webcam or iPhone
2. Sends the image directly to Gemini for analysis
3. Uses Gemini to extract text, identify sender, amount due, etc.
4. Logs all captured mail to a CSV file

The advantage of this approach is significantly improved accuracy through
direct image understanding rather than a two-step OCR + text analysis process.
"""

import cv2
import csv
import yaml
import os
import sys
import time
import base64
import json
import re
import platform
import textwrap
import tempfile
from pathlib import Path
from dotenv import load_dotenv
import numpy as np
import google.generativeai as genai
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress
from rich.table import Table
from datetime import datetime

# Load environment variables and configure Gemini
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

# Constants
LOG_DIR = Path("mail_data")
LOG_FILE = LOG_DIR / "mail_log.csv"
IMAGES_DIR = LOG_DIR / "images"
console = Console()

# Configure the Gemini models - multiple options for different contexts
MODELS = {
    "primary": "gemini-1.5-pro",     # Most reliable for image analysis
    "backup": "gemini-1.5-flash",    # Backup model (previously used gemini-pro-vision but it was deprecated)
    "flash": "gemini-1.5-flash"      # Fast model for quick analysis
}

# Error log file
ERROR_LOG = LOG_DIR / "error_log.json"

# Ensure directories exist
LOG_DIR.mkdir(exist_ok=True)
IMAGES_DIR.mkdir(exist_ok=True)

def clean_and_parse_yaml(raw_text):
    """Clean Gemini response and parse YAML safely
    
    This handles common issues like triple backticks, YAML markers,
    and other text that might be included in Gemini responses.
    It also handles multiple YAML documents separated by --- markers.
    
    Returns:
    - For single documents: a single dictionary
    - For multi-page documents: a list of dictionaries, one per document
    """
    # Start with cleaning the text
    cleaned = raw_text.strip()
    
    # Remove YAML markers if present
    if cleaned.startswith("```yaml"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```YAML"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
        
    # Remove closing backticks
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    
    # Additional cleanup for common issues
    cleaned = cleaned.strip()
    
    # Try to find the start of YAML content if not at beginning
    if not cleaned.startswith("Sender:"):
        yaml_start = cleaned.find("Sender:")
        if yaml_start != -1:
            cleaned = cleaned[yaml_start:]
    
    # Parse the cleaned YAML - handle multiple documents
    try:
        # Use safe_load_all to handle multiple YAML documents separated by ---
        documents = list(yaml.safe_load_all(cleaned))
        
        # If no documents were found
        if not documents:
            raise Exception("No valid YAML documents found")
        
        # Verify each document is a dictionary
        for doc in documents:
            if not isinstance(doc, dict):
                raise Exception(f"Document is not a valid YAML dictionary: {doc}")
        
        # For multi-page documents, return the full list of documents
        if len(documents) > 1:
            return documents
        # For single documents, return just the document (not in a list)
        else:
            return documents[0]
    except yaml.YAMLError as err:
        # Re-raise with more context
        raise yaml.YAMLError(f"YAML parsing error after cleaning: {err}. Content was: {cleaned[:100]}...")

class MailDocument:
    """Class to handle multi-page documents"""
    
    def __init__(self):
        self.pages = []
        self.images = []
        self.summary = None
        self.quick_scan_result = None
    
    def add_page(self, image, snapshot_time=None, quick_scan=None):
        """Add a page image to the document"""
        if snapshot_time is None:
            snapshot_time = datetime.now()
            
        self.pages.append({
            'image': image,
            'time': snapshot_time,
            'quick_scan': quick_scan
        })
        self.images.append(image)
        
        # Store the quick scan result of the first page for potential fallback
        if self.quick_scan_result is None and quick_scan:
            self.quick_scan_result = quick_scan
    
    def get_summary(self):
        """Get a summary from Gemini"""
        if not self.images:
            return None
        
        if len(self.images) == 1:
            # Single page document
            self.summary = analyze_mail_image(self.images[0], self.quick_scan_result)
        else:
            # Multi-page document
            self.summary = analyze_multipage_mail(self.images, self.quick_scan_result)
            
        return self.summary

def encode_image(image):
    """Convert image to base64 for Gemini API"""
    # Convert the image to JPEG format
    success, encoded_image = cv2.imencode('.jpg', image)
    if not success:
        return None
    
    # Convert to base64
    return base64.b64encode(encoded_image).decode('utf-8')

def log_error(error_type, error_msg, raw_text=None, doc_id=None):
    """Log errors to JSON file for debugging and troubleshooting"""
    error_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "error_type": error_type,
        "error_message": str(error_msg),
        "doc_id": doc_id or f"unknown_{int(time.time())}"
    }
    
    # Include raw text for YAML parsing errors (truncated for size)
    if raw_text:
        if len(raw_text) > 500:
            error_data["raw_text"] = raw_text[:500] + "... [truncated]"
        else:
            error_data["raw_text"] = raw_text
    
    try:
        # Read existing errors if file exists
        if ERROR_LOG.exists():
            with ERROR_LOG.open('r') as f:
                try:
                    errors = json.load(f)
                except json.JSONDecodeError:
                    errors = []
        else:
            errors = []
        
        # Add new error
        errors.append(error_data)
        
        # Write back to file
        with ERROR_LOG.open('w') as f:
            json.dump(errors, f, indent=2)
            
        console.print(f"[yellow]Error logged to {ERROR_LOG}[/]")
    except Exception as e:
        console.print(f"[bold red]Failed to log error: {str(e)}[/]")
    
    return error_data

def extract_quick_scan_data(quick_scan_text):
    """Extract structured data from quick scan results"""
    if not quick_scan_text:
        return None
    
    data = {
        "Sender": "Unknown",
        "DocumentDate": "Unknown",
        "Category": "other",
        "AmountDue": "Unknown",
        "DueDate": "Unknown",
        "ActionNeeded": "review",
        "Overview": "Generated from quick scan"
    }
    
    # Look for company/sender name
    sender_patterns = [
        r"(?:sent by|from|sender)[:\s]+([A-Za-z0-9\s&.,]+)",
        r"([A-Za-z0-9\s&.,]+)(?:bill|statement|letter)",
    ]
    
    for pattern in sender_patterns:
        match = re.search(pattern, quick_scan_text, re.IGNORECASE)
        if match:
            data["Sender"] = match.group(1).strip()
            break
    
    # Look for document type
    type_match = re.search(r"(?:type|is a|document is)[:\s]+(bill|statement|letter|invoice|notice|ad)", 
                          quick_scan_text, re.IGNORECASE)
    if type_match:
        doc_type = type_match.group(1).lower()
        if doc_type in ["bill", "invoice"]:
            data["Category"] = "bill"
            data["ActionNeeded"] = "pay"
        elif doc_type == "statement":
            data["Category"] = "statement"
            data["ActionNeeded"] = "file"
        else:
            data["Category"] = doc_type
    
    # Look for amounts
    amount_match = re.search(r'[\$£€](\d+(?:,\d+)*(?:\.\d+)?)', quick_scan_text)
    if amount_match:
        data["AmountDue"] = "$" + amount_match.group(1)
    
    # Look for dates
    date_match = re.search(r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})', quick_scan_text)
    if date_match:
        data["DueDate"] = date_match.group(1)
    
    # Add timestamp
    data['ProcessedDate'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data['QuickScanFallback'] = True
    
    return data

def analyze_mail_image(image, quick_scan_result=None):
    """Analyze mail image directly with Gemini vision capabilities"""
    console.print("[bold]Analyzing image with Gemini...[/]")
    
    # Encode the image once for all attempts
    base64_image = encode_image(image)
    if not base64_image:
        console.print("[bold red]Failed to encode image[/]")
        return get_default_data("Failed to encode image")
    
    # Define the prompt for mail analysis
    prompt = """
    Analyze this image of a physical mail/letter. Extract the following information as accurately as possible:
    - Sender: Who sent this mail/letter? Look for logos, letterhead, return address, or signatures.
    - DocumentDate: What is the date on the document? Format as MM/DD/YYYY if possible.
    - Category: Classify as bill, statement, personal, ad, legal, or other.
    - AmountDue: If this is a bill, what amount is due? Extract the exact currency and amount.
    - DueDate: If there's a payment due, when is it due? Format as MM/DD/YYYY if possible.
    - ActionNeeded: What action is required? (pay, reply, file, read, none)
    - Overview: Provide a 1-2 sentence summary of the mail's content and purpose.
    
    Present the information as YAML format ONLY. 
    If you can't determine some information, use "Unknown" or "N/A".
    DO NOT include anything but the YAML document.
    """
    
    # Create content parts including both the text prompt and the image
    content_parts = [
        {"text": prompt},
        {"inline_data": {"mime_type": "image/jpeg", "data": base64_image}}
    ]
    
    # Try primary model first, then fall back to backup if needed
    model_attempts = ["primary", "backup"]
    raw_response = None
    
    for attempt, model_key in enumerate(model_attempts):
        model_name = MODELS[model_key]
        try:
            console.print(f"[yellow]Attempt {attempt+1}/{len(model_attempts)}: Using model {model_name}...[/]")
            
            # Initialize the model
            model = genai.GenerativeModel(model_name)
            
            # Set safety settings to be more permissive
            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
            ]
            
            # Generate content with the image
            response = model.generate_content(
                content_parts,
                safety_settings=safety_settings,
                generation_config={"temperature": 0.1, "top_p": 0.8, "top_k": 40}
            )
            
            # Check if we got a response
            if not response.text:
                raise Exception("Empty response received")
                
            # Store raw response text
            raw_response = response.text
            console.print(f"[dim]Raw response: {raw_response}[/]")
            
            try:
                # Use our improved YAML cleaning and parsing function
                data = clean_and_parse_yaml(raw_response)
                
                # Add capture timestamp
                data['ProcessedDate'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                console.print("[bold green]Successfully extracted information![/]")
                return data
                
            except yaml.YAMLError as yaml_err:
                console.print(f"[red]YAML parsing error: {yaml_err}[/]")
                # Log the full error with the raw text
                log_error("YAML_PARSING_ERROR", yaml_err, raw_response)
                raise Exception(f"Failed to parse YAML: {yaml_err}")
            
        except Exception as e:
            console.print(f"[bold red]Error with model {model_name}:[/] {str(e)}")
            if attempt < len(model_attempts) - 1:
                console.print(f"[yellow]Falling back to next model...[/]")
            else:
                console.print(f"[red]All model attempts failed.[/]")
                
                # Log the final failure
                error_data = log_error("ALL_MODELS_FAILED", str(e), raw_response)
                
                # Try to extract information from quick scan if available
                if quick_scan_result:
                    console.print("[yellow]Attempting to extract data from quick scan...[/]")
                    quick_data = extract_quick_scan_data(quick_scan_result)
                    if quick_data:
                        console.print("[green]Successfully extracted basic information from quick scan![/]")
                        return quick_data
                
                return get_default_data(f"All models failed: {str(e)}")

def get_default_data(error_msg):
    """Return default data structure when analysis fails"""
    return {
        "Sender": "ERROR - See log",
        "DocumentDate": "Unknown",
        "Category": "other",
        "AmountDue": "Unknown",
        "DueDate": "Unknown",
        "ActionNeeded": "review",
        "Overview": f"Error during analysis: {error_msg}",
        "ProcessedDate": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "AnalysisError": error_msg
    }

def analyze_multipage_mail(images, quick_scan_result=None):
    """Analyze multiple pages of mail with Gemini"""
    console.print(f"[bold]Analyzing {len(images)} pages with Gemini...[/]")
    
    # Encode all images first
    base64_images = []
    for img in images:
        encoded = encode_image(img)
        if encoded:
            base64_images.append(encoded)
    
    if not base64_images:
        console.print("[bold red]Failed to encode any images[/]")
        return get_default_data("Failed to encode images") 
    
    # Define the prompt for multi-page mail analysis
    prompt = f"""
    Analyze these {len(images)} pages of a physical mail/letter. They are multiple pages of the same document.
    Extract the following information as accurately as possible:
    - Sender: Who sent this mail/letter? Look for logos, letterhead, return address, or signatures.
    - DocumentDate: What is the date on the document? Format as MM/DD/YYYY if possible.
    - Category: Classify as bill, statement, personal, ad, legal, or other.
    - AmountDue: If this is a bill, what amount is due? Extract the exact currency and amount.
    - DueDate: If there's a payment due, when is it due? Format as MM/DD/YYYY if possible.
    - ActionNeeded: What action is required? (pay, reply, file, read, none)
    - Overview: Provide a 2-3 sentence summary of the mail's content and purpose.
    
    Present the information as YAML format ONLY.
    If you can't determine some information, use "Unknown" or "N/A".
    DO NOT include anything but the YAML document.
    """
    
    # Create content parts with the prompt and all images
    content_parts = [{"text": prompt}]
    
    # Add each image as a part
    for b64_img in base64_images:
        content_parts.append({
            "inline_data": {
                "mime_type": "image/jpeg",
                "data": b64_img
            }
        })
    
    # Try primary model first, then fall back to backup if needed
    model_attempts = ["primary", "backup"]
    raw_response = None
    
    for attempt, model_key in enumerate(model_attempts):
        model_name = MODELS[model_key]
        try:
            console.print(f"[yellow]Attempt {attempt+1}/{len(model_attempts)}: Using model {model_name} for multi-page analysis...[/]")
            
            # Initialize the model
            model = genai.GenerativeModel(model_name)
            
            # Set safety settings to be more permissive
            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
            ]
            
            # Generate content with the images
            response = model.generate_content(
                content_parts,
                safety_settings=safety_settings,
                generation_config={"temperature": 0.1, "top_p": 0.8, "top_k": 40}
            )
            
            # Check if we got a response
            if not response.text:
                raise Exception("Empty response received")
                
            # Store raw response
            raw_response = response.text
            console.print(f"[dim]Raw multi-page response: {raw_response}[/]")
            
            try:
                # Use our improved YAML cleaning and parsing function
                data = clean_and_parse_yaml(raw_response)
                
                # Handle case where multiple YAML documents were returned
                if isinstance(data, list):
                    console.print(f"[yellow]Multiple document records found ({len(data)}). Merging data...[/]")
                    
                    # Start with the first document as our base
                    merged_data = data[0].copy()
                    
                    # Merge in data from other documents, preferring non-Unknown values
                    for doc in data[1:]:
                        for key, value in doc.items():
                            # Only overwrite if current value is unknown/empty and new value isn't
                            if (key not in merged_data or 
                                merged_data[key] in ["Unknown", "N/A", "", None] and 
                                value not in ["Unknown", "N/A", "", None]):
                                merged_data[key] = value
                    
                    # Use the merged data
                    data = merged_data
                
                # Add capture timestamp and page count
                data['ProcessedDate'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                data['PageCount'] = len(images)
                data['MultiDocumentFormat'] = True  # Flag that this came from multiple YAML docs
                
                console.print("[bold green]Successfully extracted information from multi-page document![/]")
                return data
                
            except yaml.YAMLError as yaml_err:
                console.print(f"[red]YAML parsing error: {yaml_err}[/]")
                # Log the full error with the raw text
                log_error("YAML_PARSING_ERROR_MULTIPAGE", yaml_err, raw_response)
                raise Exception(f"Failed to parse YAML: {yaml_err}")
            
        except Exception as e:
            console.print(f"[bold red]Error with model {model_name}:[/] {str(e)}")
            if attempt < len(model_attempts) - 1:
                console.print(f"[yellow]Falling back to next model...[/]")
            else:
                console.print(f"[red]All model attempts failed.[/]")
                
                # Log the final failure
                error_data = log_error("ALL_MODELS_FAILED_MULTIPAGE", str(e), raw_response)
                
                # Try to extract information from quick scan if available for the first page
                if quick_scan_result:
                    console.print("[yellow]Attempting to extract data from quick scan of first page...[/]")
                    quick_data = extract_quick_scan_data(quick_scan_result)
                    if quick_data:
                        # Add page count to quick scan data
                        quick_data['PageCount'] = len(images)
                        quick_data['QuickScanFallback'] = True
                        quick_data['MultiPageDocument'] = True
                        console.print("[green]Successfully extracted basic information from quick scan![/]")
                        return quick_data
                
                return get_default_data_multipage(f"All models failed: {str(e)}", len(images))

def get_default_data_multipage(error_msg, page_count):
    """Return default data structure for multi-page documents when analysis fails"""
    default_data = get_default_data(error_msg)
    default_data['PageCount'] = page_count
    return default_data

def append(row: dict):
    """Append a row to the CSV log file"""
    first = not LOG_FILE.exists()
    with LOG_FILE.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if first: w.writeheader()
        w.writerow(row)

def save_images(images, doc_id):
    """Save images to disk with a document ID"""
    image_paths = []
    for i, img in enumerate(images):
        # Create a filename with document ID and page number
        filename = f"{doc_id}_page{i+1}.jpg"
        filepath = IMAGES_DIR / filename
        
        # Save the image
        cv2.imwrite(str(filepath), img)
        image_paths.append(str(filepath))
        
        console.print(f"[green]Saved image to {filepath}[/]")
    
    return image_paths

def find_document_corners(image):
    """Find and return the corners of a document in the image"""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply Canny edge detection
    edges = cv2.Canny(blur, 75, 200)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the largest contour by area
    largest_contour = max(contours, key=cv2.contourArea, default=None)
    
    if largest_contour is None:
        return None
    
    # Approximate the contour to get a polygon
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    # If we have a quadrilateral, return its corners
    if len(approx) == 4:
        return approx.reshape(4, 2)
    
    # If no suitable quadrilateral found, return None
    return None

def order_points(pts):
    """Order points in top-left, top-right, bottom-right, bottom-left order"""
    # Initialize a list of coordinates that will be ordered
    rect = np.zeros((4, 2), dtype=np.float32)
    
    # The top-left point will have the smallest sum
    # The bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    # Compute the difference between the points
    # The top-right point will have the smallest difference
    # The bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect

def four_point_transform(image, pts):
    """Apply a perspective transform to obtain a top-down view of the document"""
    # Obtain a consistent order of the points
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    # Compute the width of the new image
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    # Compute the height of the new image
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    # Construct the set of destination points
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype=np.float32)
    
    # Compute the perspective transform matrix and apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
    return warped

def speak(text):
    """Use macOS say command to speak text"""
    os.system(f'say "{text}"')

def display_help(console):
    """Display help information"""
    table = Table(title="Show-and-Tell Mail Controls")
    table.add_column("Key", style="cyan")
    table.add_column("Action", style="green")
    
    table.add_row("SPACE", "Capture current frame")
    table.add_row("P", "Add additional page to current document")
    table.add_row("N", "Finish current document and start a new one")
    table.add_row("M", "Toggle mirror mode (flips camera horizontally)")
    table.add_row("H", "Show this help")
    table.add_row("R", "Reconnect to camera")
    table.add_row("ESC", "Quit application")
    
    console.print(table)

def display_summary(data, console):
    """Display a rich formatted summary of the mail data"""
    if not data:
        console.print("[bold red]No data available to display[/]")
        return
    
    # Create a fancy panel for the sender and overview
    console.print(Panel(
        f"[bold green]{data.get('Sender', 'Unknown Sender')}[/]\n[italic]{data.get('Overview', 'No overview available')}[/]",
        title="Mail Summary",
        border_style="green"
    ))
    
    # Create a detailed table for all the other fields
    table = Table(title="Mail Details")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="green")
    
    # Add all fields except Sender and Overview which are already in the panel
    for key, value in data.items():
        if key not in ['Sender', 'Overview']:
            table.add_row(key, str(value))
    
    console.print(table)

def find_available_cameras():
    """Find all available cameras on the system"""
    console.print("[bold]Scanning for camera devices...[/]")
    
    available_cameras = []
    max_cameras_to_check = 10
    
    # On macOS, check if we can try to directly get continuity camera
    system = platform.system()
    
    # Suppress OpenCV warnings during camera scanning
    original_stderr = sys.stderr
    try:
        with open(os.devnull, 'w') as null_stream:
            sys.stderr = null_stream
            
            for i in range(max_cameras_to_check):
                try:
                    cap = cv2.VideoCapture(i)
                    if cap.isOpened():
                        ret, frame = cap.read()
                        if ret:
                            # Get camera name if possible
                            name = f"Camera #{i}"
                            is_likely_iphone = False
                            
                            # Try to get more info about the camera
                            if hasattr(cap, 'getBackendName'):
                                backend = cap.getBackendName()
                                name = f"Camera #{i} ({backend})"
                                
                                # On macOS, add hint for Continuity Camera
                                if system == "Darwin" and "AVFOUNDATION" in backend and i > 0:
                                    name += " (Likely iPhone)"
                                    is_likely_iphone = True
                            
                            # Try to check resolution - iPhones have higher resolution
                            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                            
                            if width > 1280 or height > 720:  # Higher than HD
                                is_likely_iphone = True
                                name += f" ({width}x{height})"
                            
                            # Add to candidates, with likely iPhones first
                            if is_likely_iphone:
                                available_cameras.insert(0, (i, name))
                            else:
                                available_cameras.append((i, name))
                    cap.release()
                except Exception:
                    pass
    finally:
        # Restore stderr
        sys.stderr = original_stderr
    
    return available_cameras

def connect_to_camera(camera_index):
    """Connect to a camera and verify it's working"""
    console.print(f"[bold]Connecting to camera #{camera_index}...[/]")
    
    # Initialize camera with higher resolution settings
    cam = cv2.VideoCapture(camera_index)
    
    # Try to set higher resolution for iPhone
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    # Verify connection
    if not cam.isOpened():
        console.print("[bold red]Failed to connect![/]")
        return None
    
    # Verify we can read frames
    ret, frame = cam.read()
    if not ret or frame is None:
        console.print("[bold red]Connected but can't read frames![/]")
        cam.release()
        return None
    
    # Success
    width = cam.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
    console.print(f"[bold green]Successfully connected! Resolution: {width}x{height}[/]")
    
    return cam

def reconnect_camera(camera_index):
    """Try to reconnect to the camera if connection is lost"""
    for attempt in range(3):
        console.print(f"[yellow]Reconnection attempt {attempt+1}/3...[/]")
        cam = connect_to_camera(camera_index)
        if cam is not None:
            return cam
        time.sleep(1)  # Wait briefly between attempts
    
    console.print("[bold red]Failed to reconnect after multiple attempts.[/]")
    return None

def main():
    """Main application function"""
    console.print(Panel(
        f"[bold green]Gemini Vision Mail Processor[/]\n"
        f"Primary model: {MODELS['primary']}\n"
        f"Backup model: {MODELS['backup']}\n"
        f"This version uses Gemini's vision capabilities to directly analyze mail.",
        title="Show-and-Tell Mail"
    ))
    
    # Check for Continuity Camera support via Info.plist
    if os.path.exists("Info.plist"):
        console.print("[bold green]Info.plist detected![/] iPhone Continuity Camera should be supported.")
    else:
        console.print("[yellow]Info.plist not found. iPhone Continuity Camera may not work properly.[/]")
        console.print("[yellow]If using an iPhone as webcam, run from the directory with Info.plist.[/]")
    
    # Look for available cameras
    console.print("[bold]Searching for cameras...[/]")
    camera_options = find_available_cameras()
    
    if not camera_options:
        console.print("[bold red]No cameras detected![/]")
        console.print("[yellow]Please ensure your camera is connected and permissions are granted.[/]")
        return
    
    # Choose a camera
    console.print(f"[bold green]{len(camera_options)} camera(s) found:[/]")
    for idx, (cam_id, name) in enumerate(camera_options):
        console.print(f"  {idx+1}. {name}")
    
    # If we found multiple cameras, let user choose
    camera_index = camera_options[0][0]  # Default to first option
    try:
        from rich.prompt import Prompt
        selection = Prompt.ask(
            "Select camera to use",
            choices=[str(i+1) for i in range(len(camera_options))],
            default="1"
        )
        camera_index = camera_options[int(selection)-1][0]
    except (ImportError, KeyboardInterrupt, EOFError):
        console.print(f"Using first camera option: {camera_options[0][1]}")
    
    # Connect to the selected camera
    cam = connect_to_camera(camera_index)
    if cam is None:
        console.print("[bold red]Failed to initialize camera. Please try again.[/]")
        return
    
    # Display startup message
    console.print(Panel(
        "[bold green]Camera ready.[/]\n"
        "Point a letter at the camera and press SPACE to capture.\n"
        "Press P to add additional pages to a multi-page document.",
        title="Show-and-Tell Mail",
        subtitle="ESC to quit, H for help"
    ))
    
    # Set up document state
    current_document = MailDocument()
    capturing_multipage = False
    mirror_display = True  # Default to mirrored display for easier positioning
    
    # Main loop
    while True:
        # Read frame
        ret, frame = cam.read()
        
        # Handle connection loss
        if not ret or frame is None:
            console.print("[bold red]Camera connection lost![/]")
            console.print("[yellow]Press R to attempt reconnection or ESC to quit.[/]")
            
            # Wait for user input
            while True:
                key = cv2.waitKey(100) & 0xFF
                if key == 27:  # ESC
                    break
                elif key == ord('r') or key == ord('R'):
                    # Try to reconnect
                    cam.release()
                    cam = reconnect_camera(camera_index)
                    if cam is not None:
                        console.print("[bold green]Reconnected successfully![/]")
                        break
                    else:
                        console.print("[bold red]Reconnection failed. Press ESC to quit or R to try again.[/]")
            
            # If we couldn't reconnect and user pressed ESC, exit
            if not cam or not cam.isOpened():
                break
            continue
        
        # Draw a rectangle to guide document placement
        height, width = frame.shape[:2]
        cv2.rectangle(frame, (width//6, height//6), (width*5//6, height*5//6), (0, 255, 0), 2)
        
        # Show status text
        status_text = "CAPTURING MULTI-PAGE DOCUMENT" if capturing_multipage else "READY TO CAPTURE"
        cv2.putText(frame, status_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "SPACE=capture  P=add page  N=new doc  H=help  R=reconnect  ESC=quit", 
                   (20, height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Mirror the frame if needed
        display_frame = cv2.flip(frame, 1) if mirror_display else frame.copy()
        
        # Add mirroring indicator
        mirror_status = "MIRROR ON" if mirror_display else "MIRROR OFF"
        cv2.putText(display_frame, mirror_status, (width - 150, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Display the frame
        cv2.imshow("Mail Vision (Gemini)", display_frame)
        k = cv2.waitKey(1) & 0xFF
        
        if k == 27:  # ESC
            break
        
        elif k == 32:  # SPACE
            console.print("[yellow]⏳ Capturing…[/]")
            
            # Capture the full frame (no cropping - more reliable for Gemini)
            captured_image = frame.copy()
            
            if not capturing_multipage:
                # Start a new document
                current_document = MailDocument()
                capturing_multipage = True
                
            # Add the page to the current document
            current_document.add_page(captured_image)
            page_num = len(current_document.pages)
            console.print(f"[green]✓ Page {page_num} captured[/]")
            
            # Quick analysis of what was captured
            console.print("[yellow]Performing quick scan of image content...[/]")
            
            # Create a simple prompt for quick analysis
            try:
                # Encode the image
                base64_image = encode_image(captured_image)
                if base64_image:
                    # Use a more reliable model for quick scan
                    # Try primary model first, then fallback to flash for speed
                    quick_models = [MODELS["primary"], MODELS["flash"]]
                    quick_summary = None
                    
                    for model_name in quick_models:
                        try:
                            console.print(f"[yellow]Trying {model_name} for quick analysis...[/]")
                            
                            # Initialize the model
                            model = genai.GenerativeModel(model_name)
                            
                            # Define a specific prompt to extract key details
                            prompt = """
                            Look at this image of a mail/document and extract ONLY:
                            1. Who sent it (company or person name)
                            2. What type of document it is (bill, statement, letter, etc.)
                            3. Any prominent dollar amounts or dates visible
                            
                            Be very concise - just the facts you can clearly see.
                            If you can't determine something, just skip it.
                            """
                            
                            # Create content parts
                            content_parts = [
                                {"text": prompt},
                                {"inline_data": {"mime_type": "image/jpeg", "data": base64_image}}
                            ]
                            
                            # Set low temperature for factual response
                            generation_config = {
                                "temperature": 0.1,
                                "top_p": 0.8,
                                "top_k": 40,
                                "max_output_tokens": 100  # Keep it brief
                            }
                            
                            # Generate quick content summary
                            response = model.generate_content(
                                content_parts,
                                generation_config=generation_config
                            )
                            
                            if response.text and len(response.text.strip()) > 10:
                                quick_summary = response.text.strip()
                                break
                            else:
                                raise Exception("Empty or too short response")
                                
                        except Exception as model_error:
                            console.print(f"[yellow]Model {model_name} failed: {str(model_error)}[/]")
                            continue
                    
                    # Display and speak if we got a summary
                    if quick_summary:
                        console.print(f"[cyan]Quick scan result:[/] {quick_summary}")
                        speak(f"Page {page_num} captured: {quick_summary}")
                        
                        # Update the document page with the quick scan result
                        if page_num > 0 and page_num <= len(current_document.pages):
                            current_document.pages[page_num-1]['quick_scan'] = quick_summary
                            
                            # Store as fallback for the document
                            if page_num == 1:  # If this is the first page
                                current_document.quick_scan_result = quick_summary
                                console.print("[dim]Stored quick scan as fallback data source[/]")
                    else:
                        console.print("[yellow]Could not get quick summary from any model[/]")
                        speak(f"Page {page_num} captured")
                else:
                    speak(f"Page {page_num} captured")
            except Exception as e:
                # Fall back to basic notification if quick scan fails
                console.print(f"[yellow]Quick scan unavailable: {str(e)}[/]")
                speak(f"Page {page_num} captured")
        
        elif k == ord('p') or k == ord('P'):  # P key for multiple pages
            if capturing_multipage:
                console.print("[yellow]Ready for next page. Press SPACE to capture.[/]")
            else:
                console.print("[yellow]No active document. Press SPACE to start a new document.[/]")
        
        elif k == ord('n') or k == ord('N'):  # N key to finish the document
            if capturing_multipage and current_document.pages:
                # Process the complete document
                console.print("[yellow]⏳ Processing document with Gemini Vision...[/]")
                with Progress() as progress:
                    task = progress.add_task("[cyan]Analyzing with Gemini...", total=1)
                    data = current_document.get_summary()
                    progress.update(task, completed=1)
                
                if data:
                    # Generate a unique document ID using timestamp and first letters of sender
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    sender_init = "".join([c for c in data.get('Sender', 'unknown')[:3] if c.isalpha()]).upper()
                    doc_id = f"{timestamp}_{sender_init}"
                    
                    # Save images to disk
                    console.print("[yellow]Saving images to disk...[/]")
                    image_paths = save_images(current_document.images, doc_id)
                    
                    # Add image paths to the data
                    data['ImagePaths'] = ';'.join(image_paths)
                    data['DocumentID'] = doc_id
                    
                    # Save to CSV
                    append(data)
                    
                    # Display summary
                    display_summary(data, console)
                    
                    # Show where files are saved
                    console.print(f"[bold green]Document saved![/]")
                    console.print(f"[green]Images: {IMAGES_DIR}[/]")
                    console.print(f"[green]Log: {LOG_FILE}[/]")
                    
                    # Speak a summary
                    sender = data.get('Sender', 'unknown sender')
                    action_needed = data.get('ActionNeeded', 'unknown action')
                    amount_due = data.get('AmountDue', 'no amount')
                    due_date = data.get('DueDate', '')
                    
                    speech_text = f"Letter from {sender} needs {action_needed}"
                    if amount_due not in ['', 'N/A', 'Unknown', None, 'no amount']:
                        speech_text += f" with amount {amount_due}"
                    if due_date and due_date not in ['', 'N/A', 'Unknown']:
                        speech_text += f" by {due_date}"
                        
                    speak(speech_text)
                
                # Reset for next document
                capturing_multipage = False
                console.print("[green]✓ Document processed and logged[/]")
            else:
                console.print("[yellow]No active document to process.[/]")
        
        elif k == ord('h') or k == ord('H'):  # H key for help
            display_help(console)
            
        elif k == ord('m') or k == ord('M'):  # M key to toggle mirroring
            mirror_display = not mirror_display
            mirror_status = "ON" if mirror_display else "OFF"
            console.print(f"[cyan]Mirror mode: {mirror_status}[/]")
        
        elif k == ord('r') or k == ord('R'):  # R key to reconnect
            console.print("[yellow]Attempting to reconnect camera...[/]")
            cam.release()
            cam = reconnect_camera(camera_index)
            if cam is None:
                console.print("[bold red]Reconnection failed. Press ESC to quit or R to try again.[/]")
    
    # Clean up
    if cam is not None:
        cam.release()
    cv2.destroyAllWindows()
    console.print("[bold green]Session complete.[/]")
    console.print(f"[green]Mail log saved to: {LOG_FILE}[/]")
    console.print(f"[green]Images saved in: {IMAGES_DIR}[/]")

if __name__ == "__main__":
    main()
