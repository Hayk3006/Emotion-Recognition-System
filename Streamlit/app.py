import os
import json
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import streamlit as st

# Load credentials JSON content from environment variable
credentials_content = os.getenv('GOOGLE_DRIVE_CREDENTIALS', '{}')
try:
    credentials_json = json.loads(credentials_content)
except json.JSONDecodeError:
    st.error("Error: Invalid credentials JSON format. Check your credentials environment variable.")
    st.stop()

# Save the JSON to a file if necessary
credentials_file = 'credentials.json'
with open(credentials_file, 'w') as f:
    json.dump(credentials_json, f)

# Authenticate and initialize PyDrive
gauth = GoogleAuth()
try:
    gauth.LoadCredentialsFile(credentials_file)
    if not gauth.credentials:
        gauth.LocalWebserverAuth()  # Opens a browser for authentication if needed
    else:
        gauth.Authorize()
except Exception as e:
    st.error(f"Authentication Error: {e}")
    st.stop()

drive = GoogleDrive(gauth)

# Google Drive folder ID (replace with your specific folder ID)
folder_id = "1ybZf0-PN3AmC39GnaPJGAalX0f9ha2IS"
query = f"'{folder_id}' in parents and trashed=false"
try:
    file_list = drive.ListFile({'q': query}).GetList()
except Exception as e:
    st.error(f"Error accessing Google Drive folder: {e}")
    st.stop()

# Display files in Streamlit
st.write(f"Files in Google Drive Folder ID: {folder_id}")
for file in file_list:
    st.write(f"{file['title']} ({file['id']})")
