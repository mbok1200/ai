# config/GoogleDrive.py

from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import pickle
import os
from dotenv import load_dotenv
import json
from googleapiclient.http import MediaIoBaseDownload

from helpers.helpers_fn import sanitize_filename

load_dotenv()  # take environment variables from .env.

SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
gdrive = "data/gdrive"
cred_path = os.path.join(BASE_DIR, "credentials.json")
first_folder_id = os.environ.get('FIRST_FOLDER_ID')
file_map = {}
    
# Authenticate and create the service
def get_gdrive_service():
    creds = None
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(cred_path, SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
    service = build('drive', 'v3', credentials=creds)
    return service
def process_folder(service, folder_id, local_path=""):
    query = f"'{folder_id}' in parents and trashed=false"
    results = service.files().list(
        q=query,
        fields="nextPageToken, files(id, name, mimeType, webViewLink)"
    ).execute()
    items = results.get('files', [])
    for item in items:
        safe_name = sanitize_filename(item['name'])
        if local_path == "":
            local_path = os.path.join(gdrive, local_path)
        item_path = os.path.join(local_path, safe_name)

        if item['mimeType'] == 'application/vnd.google-apps.folder':
            os.makedirs(item_path, exist_ok=True)
            print(f"Entering folder: {item_path}")
            process_folder(service, item['id'], item_path)
        else:
            print(f"Processing file: {item_path}")
            download_file(service, item['id'], item_path, item['mimeType'])
            file_map[sanitize_filename(item['name'])] = item["webViewLink"]
def download_file(service, file_id, file_name, mime_type):
    request = None

    if mime_type.startswith('application/vnd.google-apps.'):
        if mime_type == 'application/vnd.google-apps.document':
            export_mime = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
            file_name = file_name.rsplit('.', 1)[0] + '.docx'  # Зміна розширення
        elif mime_type == 'application/vnd.google-apps.spreadsheet':
            export_mime = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            file_name = file_name.rsplit('.', 1)[0] + '.xlsx'
        elif mime_type == 'application/vnd.google-apps.presentation':
            export_mime = 'application/pdf'
            file_name = file_name.rsplit('.', 1)[0] + '.pdf'
        else:
            raise Exception(f"Unsupported Google Docs mime type: {mime_type}")

        request = service.files().export_media(fileId=file_id, mimeType=export_mime)
    else:
        request = service.files().get_media(fileId=file_id)
    with open(file_name, 'wb') as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
            print(f"Download {int(status.progress() * 100)}%.")
    fh.close()
def create_dataset(file_map, output_path):
    dataset = []
    for name, url in file_map.items():
        item = {
            "instruction": "Проаналізуй цей запит",
            "input": name,
            "output": f"Посилання на документ: {url}",
            "metadata": {
                "language": "ua",
                "source": url,
                "category": name
            }
        }
        dataset.append(item)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
if __name__ == '__main__':
    service = get_gdrive_service()
    process_folder(service, first_folder_id)
    map_path = os.path.join(BASE_DIR, "gdrive_file_map.json")
    with open(map_path, "w", encoding="utf-8") as f:
        json.dump(file_map, f, ensure_ascii=False, indent=2)
    # Example of downloading a file
    # Створення датасету
    dataset_path = os.path.join(BASE_DIR, "gdrive_dataset.json")
    create_dataset(file_map, dataset_path)