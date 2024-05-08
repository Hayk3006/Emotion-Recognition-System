import os
import io
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# Initialize credentials and the Google Drive API service
CREDENTIALS_PATH = 'credentials.json'
credentials = service_account.Credentials.from_service_account_file(
    CREDENTIALS_PATH, scopes=["https://www.googleapis.com/auth/drive"]
)
drive_service = build('drive', 'v3', credentials=credentials)

# Function to list all files within a specified folder
def list_files_in_folder(folder_id):
    query = f"'{folder_id}' in parents and trashed=false"
    results = drive_service.files().list(q=query, fields="files(id, name)").execute()
    files = results.get('files', [])
    return files

# Function to download a file using its file ID
def download_file(file_id, dest_path):
    request = drive_service.files().get_media(fileId=file_id)
    fh = io.FileIO(dest_path, 'wb')
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
        print(f"Download {int(status.progress() * 100)}% complete.")

# Example folder ID from your provided link
folder_id = '1ybZf0-PN3AmC39GnaPJGAalX0f9ha2IS'
destination_dir = 'downloaded_models'  # Change this to your desired path

# Create the destination directory if it doesn't exist
if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)

# List all files in the folder and download each
files = list_files_in_folder(folder_id)
for file in files:
    file_id = file['id']
    file_name = file['name']
    local_path = os.path.join(destination_dir, file_name)
    print(f"Downloading {file_name} to {local_path}")
    download_file(file_id, local_path)
