import os
import datetime
UPLOAD_PATH = 'public/storage/'
UPLOAD_FOLDER = os.path.abspath(UPLOAD_PATH)

def upload_file(file):
    if file.filename == '':
        return False
    if file:
        timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        filename = f"{timestamp}-{file.filename}"
        abs_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(abs_path)
        return abs_path  # ⬅️ selalu kembalikan absolute path
    return False

def delete_file(file_path):
    if file_path and os.path.exists(file_path):
        os.remove(file_path)
        return True
    return False