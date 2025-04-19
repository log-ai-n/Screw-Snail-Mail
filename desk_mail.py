#!/usr/bin/env python3
"""
Show‑and‑Tell Mail – webcam edition
1. Opens your default camera
2. Press SPACE to capture the current frame
3. OCR → Gemini summary → CSV log + on‑screen print
Press ESC to quit.
"""

import cv2
import csv
import yaml
import os
import sys
import platform
import textwrap
import tempfile
from pathlib import Path
from dotenv import load_dotenv
import pytesseract
import google.generativeai as genai
from rich.console import Console

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

LOG = Path("mail_log.csv")
console = Console()

def summarise(text: str) -> dict:
    prompt = f"""
    Summarise this physical letter. Extract:
      Sender, DocumentDate, Category (bill|statement|personal|ad|legal|other),
      AmountDue (if any), ActionNeeded (pay|reply|file|read|none).
    Give YAML only.

    ---\n{text[:8000]}\n---
    """
    r = genai.GenerativeModel("gemini-1.0-pro-latest").generate_content(prompt)
    return yaml.safe_load(r.text)

def append(row: dict):
    first = not LOG.exists()
    with LOG.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if first: w.writeheader()
        w.writerow(row)

def ocr(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # quick contrast boost
    gray = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return pytesseract.image_to_string(gray)

def list_available_cameras():
    """List all available camera devices"""
    console.print("[bold]Scanning for camera devices...[/]")
    
    available_cameras = []
    max_cameras_to_check = 10
    
    # On macOS, check if we can try to directly get continuity camera
    system = platform.system()
    
    # Suppress OpenCV warnings during camera scanning
    # This prevents the "device out of bound" warnings from cluttering the output
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
                            if hasattr(cap, 'getBackendName'):
                                backend = cap.getBackendName()
                                name = f"Camera #{i} ({backend})"
                                
                                # On macOS, add hint for Continuity Camera
                                if system == "Darwin" and "AVFOUNDATION" in backend and i > 0:
                                    name += " (Possibly iPhone)"
                            
                            available_cameras.append((i, name))
                    cap.release()
                except Exception as e:
                    # Just skip this camera index if there's an error
                    pass
    finally:
        # Restore stderr
        sys.stderr = original_stderr
    
    return available_cameras

def main():
    # Check for Continuity Camera support
    if os.path.exists("Info.plist"):
        console.print("[bold green]Info.plist detected![/] iPhone Continuity Camera should be supported.")
    else:
        console.print("[yellow]Info.plist not found. iPhone Continuity Camera may not work properly.[/]")
        console.print("[yellow]If using an iPhone as webcam, run from the directory with Info.plist.[/]")
    
    # List available cameras
    cameras = list_available_cameras()
    
    if not cameras:
        console.print("[bold red]Error: No cameras detected![/]")
        console.print("[yellow]Tip: Check camera permissions in System Settings → Privacy → Camera[/]")
        console.print("[yellow]Allow Terminal or your code editor to access the camera.[/]")
        return
    
    console.print(f"[bold green]{len(cameras)} camera(s) found:[/]")
    for idx, (cam_id, name) in enumerate(cameras):
        console.print(f"  {idx+1}. {name}")
    
    # Select camera
    camera_index = cameras[0][0]  # Default to first camera
    if len(cameras) > 1:
        try:
            from rich.prompt import Prompt
            selection = Prompt.ask(
                "Select camera to use",
                choices=[str(i+1) for i in range(len(cameras))],
                default="1"
            )
            camera_index = cameras[int(selection)-1][0]
        except (ImportError, KeyboardInterrupt, EOFError):
            console.print(f"Using default camera #{camera_index}")
    
    # Initialize camera
    console.print(f"[bold]Initializing camera #{camera_index}...[/]")
    cam = cv2.VideoCapture(camera_index)
    
    # Check if camera opened successfully
    if not cam.isOpened():
        console.print("[bold red]Error: Could not open camera.[/]")
        console.print("[yellow]Tip: Check camera permissions in System Settings → Privacy → Camera[/]")
        console.print("[yellow]Allow Terminal or your code editor to access the camera.[/]")
        return
        
    console.print("[bold green]Camera ready.[/]  Point a letter, press SPACE.")
    
    while True:
        # Capture frame-by-frame
        ret, frame = cam.read()
        
        # If frame was not captured successfully
        if not ret or frame is None:
            console.print("[bold red]Error: Failed to capture frame.[/]")
            break
            
        # Display the frame
        cv2.imshow("MailCam – SPACE=snap  ESC=quit", frame)
        k = cv2.waitKey(1) & 0xFF
        
        if k == 27:   # ESC
            break
            
        if k == 32:   # SPACE
            console.print("[yellow]⏳ Capturing…[/]")
            text = ocr(frame)
            data = summarise(text)
            append(data)
            console.print("[green]✓ Logged:[/]", data)
            
    # Release resources
    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
