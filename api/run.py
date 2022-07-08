from app import app
from app.controler import upload_file
from app.config import configure_upload_directory

def setup():
    configure_upload_directory()

if __name__ == '__main__':
    setup()
    app.run(host='0.0.0.0', port=8081, debug=True)
