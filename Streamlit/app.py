import streamlit as st
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import os
import json

# Load credentials from environment variable
credentials_content = os.environ.get('GOOGLE_DRIVE_CREDENTIALS', '{}')
with open('credentials.json', 'w') as f:
    f.write(credentials_content)

# Authenticate and initialize PyDrive
gauth = GoogleAuth()
gauth.LoadCredentialsFile('credentials.json')
if not gauth.credentials:
    gauth.LocalWebserverAuth()  # Will open a browser for authentication
else:
    gauth.Authorize()
drive = GoogleDrive(gauth)

# Google Drive folder ID (found in the URL)
folder_id = "1ybZf0-PN3AmC39GnaPJGAalX0f9ha2IS"

# Fetch all files from the specified folder
query = f"'{folder_id}' in parents and trashed=false"
file_list = drive.ListFile({'q': query}).GetList()

# Display the files in Streamlit
st.write(f"Files in Google Drive Folder ID: {folder_id}")
for file in file_list:
    st.write(f"{file['title']} ({file['id']})")
