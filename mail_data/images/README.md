# Mail Images Directory

This directory stores captured images of processed mail documents.

## Structure

- Images are automatically saved here when you capture mail with the application
- Naming format: `[TIMESTAMP]_[SENDER_INITIALS]_page[PAGE_NUMBER].jpg`
- Example: `20250419_AMX_page1.jpg` (American Express statement, page 1)

## Privacy Note

Images stored in this directory may contain personal and sensitive information. 
When using version control:

1. This directory is included in .gitignore to prevent accidental commits of personal mail images
2. Only this README file should be committed to the repository

## Disk Space Management

If the application is used regularly, this directory may accumulate many images over time.
You can safely delete older images you no longer need to reference, but note that this will
break links in the mail_log.csv file for those entries.
