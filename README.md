# Anti-Postal: Show-and-Tell Mail Processor

A simple application that captures physical mail through your webcam, processes it via OCR and Gemini AI, and logs the important details to a spreadsheet.

"Screw Snail Mail": Because waiting for snail mail is soooo 1999. 🐌📬 This Python-powered project is like a supercharged secretary for your physical mailbox. It captures, analyzes, and organizes your mail using computer vision and AI—think of it as giving your mail the Tony Stark Jarvis treatment! 🕶️🤖

With 95.7% Python and a dash of Shell scripting flair, this tool is here to ensure your bills, postcards, and coupons get the digital VIP treatment they deserve. No more piles of paper or guessing what's in the envelope—the AI's got it handled. So, screw snail mail and let this project take your clutter and turn it into digital harmony! 🎉📄✨

## Overview

This application allows you to:
1. Point your webcam at physical mail
2. Press SPACE to capture the image
3. Process the mail using your preferred method:
   - OCR + Gemini Text Analysis (Basic & Enhanced versions)
   - Direct Gemini Vision Analysis (Vision version - most accurate)
4. Extract key details like sender, amount due, action needed
5. Log all mail to a CSV file for easy tracking

## Setup

### Prerequisites
- Python 3.x
- Homebrew (for macOS) or Tesseract installed manually on other platforms
- A Gemini API key from [Google AI Studio](https://makersuite.google.com/app/apikey)

### Automatic Installation (Recommended)

We provide a convenient setup script that handles all installation steps automatically:

```bash
./setup.sh
```

This script will:
1. Check for Python and Homebrew (on macOS)
2. Install Tesseract OCR engine
3. Create a Python virtual environment
4. Install all required packages
5. Create a template `.env` file for your Gemini API key

After running the setup script, edit the `.env` file to add your Gemini API key.

### Manual Installation

If you prefer to set up manually:

1. Create the project directory and set up a virtual environment:
   ```bash
   python3 -m venv .venv && source .venv/bin/activate
   ```

2. Install Tesseract OCR engine:
   - macOS: `brew install tesseract`
   - Ubuntu/Debian: `sudo apt-get install tesseract-ocr`
   - Fedora: `sudo dnf install tesseract`

3. Install required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Add your Gemini API key to the `.env` file:
   ```
   GEMINI_API_KEY="your-actual-api-key"
   ```

## Data Storage

All processed mail data is organized in the `mail_data` directory:

```
mail_data/
├── images/          # Contains saved images of all processed mail
│   ├── 20250419_CHA_page1.jpg
│   ├── 20250419_AME_page1.jpg
│   ├── 20250419_AME_page2.jpg
│   └── ...
└── mail_log.csv     # Log file containing all mail metadata
```

### Mail Log Structure

The mail log CSV file contains structured information for each mail item:
- Sender: Organization or person who sent the mail
- DocumentDate: Date shown on the document
- Category: Type of mail (bill, statement, personal, etc.)
- AmountDue: Payment amount required (if applicable)
- DueDate: Payment due date (if applicable)
- ActionNeeded: Required action (pay, reply, file, read, none)
- ImagePaths: File paths to stored images
- DocumentID: Unique ID for each mail item
- ProcessedDate: When the mail was processed

### Image Storage

Images are automatically saved with standardized naming:
- Format: `[TIMESTAMP]_[SENDER_INITIALS]_page[PAGE_NUMBER].jpg`
- Example: `20250419_AMX_page1.jpg` (American Express statement, page 1)

## How to Use

### Using the Run Script (Recommended)

The easiest way to run the application is to use our interactive run script:

```bash
./run.sh
```

This script will:
1. Activate the virtual environment
2. Provide a menu to choose between:
   - Testing your camera
   - Running the basic version
   - Running the enhanced version
   - Running the iPhone Continuity Camera version
   - Running the Gemini Vision version (recommended)

### Manual Execution

If you prefer to run the applications directly:

1. Ensure your virtual environment is activated:
   ```bash
   source .venv/bin/activate
   ```

2. Test your camera (recommended first step):
   ```bash
   python camera_test.py
   ```

3. Launch the basic or enhanced application:
   ```bash
   python desk_mail.py
   # or
   python desk_mail_enhanced.py
   ```

### Using the Application

1. Hold the first page flat, well-lit, about 30 cm from the camera.

2. Press SPACE to capture the current frame, which will:
   - Perform OCR on the image
   - Send the text to Gemini for analysis
   - Print a summary to the console
   - Add the entry to `mail_log.csv`

3. Move to the next piece of mail and press SPACE again.

4. Press ESC when finished. Open `mail_log.csv` in your preferred spreadsheet application for a sortable overview.

### Version Options

#### Basic Version
The basic version (`desk_mail.py`) provides essential functionality with:
- Webcam capture of mail documents
- OCR using Tesseract
- Text-based analysis using Gemini
- Simple console output
- CSV logging

#### Enhanced Version
The enhanced version (`desk_mail_enhanced.py`) includes several additional features:
- **Auto-cropping**: Detects and crops document edges for better OCR
- **Multi-page support**: 
  - Press P to add additional pages to the current document
  - Press N to finish the current document and start a new one
- **Voice feedback**: Speaks confirmation of captures and document summaries
- **Improved UI**: Better visualizations and formatting with rich text interface
- **Help menu**: Press H to see available commands

#### iPhone Continuity Camera Version
The iPhone version (`desk_mail_iphone.py`) is optimized for using your iPhone as a camera:
- Improved detection of iPhone cameras
- Higher resolution settings
- Automatic reconnection if connection is lost
- Special handling for camera selection

#### Gemini Vision Version (Recommended)
The Vision version (`desk_mail_vision.py`) represents a major improvement in accuracy:
- **Direct image analysis**: Sends images directly to Gemini's vision model
- **No OCR step**: Eliminates OCR errors by letting Gemini analyze the image directly
- **Superior understanding**: Better at extracting information from complex layouts
- **Multi-page support**: Handles multi-page documents with unified analysis
- **Enhanced output**: Provides more detailed and accurate summaries
- **Due date detection**: Additionally extracts payment due dates for bills
- **Richer context**: Includes an overview summarizing the document's purpose

This version is recommended for the best accuracy as it leverages Gemini's advanced vision capabilities, especially for documents with complex layouts, handwriting, or poor print quality.

## Tips for Better OCR

| Tip | Reason |
|-----|--------|
| Use bright, even lighting; avoid shadows | Tesseract accuracy drops with low contrast |
| Fill the camera frame; keep page square | Bigger text = fewer recognition errors |
| For glossy letters: tilt slightly to dodge glare | Prevents blown-out highlights |
| Use iPhone Continuity Camera as "webcam" if available | Sharper sensor for better results |

## Camera Setup and Testing

Before using the main application, you may want to verify that your camera is properly configured and accessible. A test script is provided for this purpose:

```bash
python camera_test.py
```

This utility will:
1. Scan for all available cameras on your system (including iPhone Continuity Camera)
2. Let you select which camera to use
3. Test if your selected webcam can be accessed
4. Display your camera feed in a window if successful
5. Provide troubleshooting steps specific to your operating system if unsuccessful

### Using iPhone as a Camera (Continuity Camera)

The application fully supports using your iPhone as a high-quality camera via Apple's Continuity Camera feature:

1. Make sure your iPhone and Mac are on the same Apple ID and have Bluetooth and WiFi enabled
2. Enable Continuity Camera on your iPhone: Settings → General → AirPlay & Handoff → Continuity Camera
3. Position your iPhone so it can see your desk (using a stand or holder is recommended)
4. Run the application from the project directory (so Info.plist is found)
5. When the camera selection prompt appears, choose your iPhone from the list

**Advantages of using iPhone camera**:
- Much higher resolution and better image quality
- Stronger autofocus capabilities
- Better performance in low light conditions
- Increased OCR accuracy due to superior optics

### Camera Permissions

For the application to work, you need to grant camera access to Terminal (or your code editor):

- **macOS**: System Settings → Privacy & Security → Camera → allow Terminal/iTerm/VS Code
- **Windows**: Settings → Privacy → Camera → ensure your terminal or editor is allowed
- **Linux**: Permissions vary by distribution (see camera_test.py for details)

**Important**: After granting permission, you may need to restart your terminal or code editor.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Camera black / permission denied | Run `python camera_test.py` to diagnose and fix camera permission issues |
| Many wrong characters (e.g., "0" vs "O") | Increase light, reshoot closer, or adjust threshold in ocr() function |
| TesseractNotFoundError | Open new terminal after `brew install tesseract` so PATH refreshes |
| Gemini quota exceeded | Summarize locally or upgrade API quota |
| "OpenCV: camera failed to properly initialize" | After granting camera permissions, restart your terminal or code editor |

## Potential Enhancements

1. Voice prompt - Add spoken confirmation after processing
   ```python
   os.system(f'say "Letter from {data["Sender"]} needs {data["ActionNeeded"]}"')
   ```

2. Auto crop - Add edge detection to isolate paper region before OCR

3. Multiple pages - Add functionality to handle multi-page documents

4. Calendar integration - Create reminders for bills with due dates

5. Improved UI - Wrap the capture loop in a PyObjC window for a better user experience

## Why These Technologies?

- opencv-python - Provides live camera feed and image processing
- tesseract + pytesseract - Converts image to text locally
- google-generativeai - Sends text to Gemini for intelligent extraction and categorization
- rich - Provides nicely formatted console output
- python-dotenv - Manages environment variables for API keys
