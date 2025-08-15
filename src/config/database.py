from flask_sqlalchemy import SQLAlchemy
from src.config.config import DATABASE_URL

# server = None  # tunnel global

# def start_ssh_tunnel():
#     global server
#     server = SSHTunnelForwarder(
#         (SSH_HOST, SSH_PORT),
#         ssh_username=SSH_USERNAME,
#         ssh_password=SSH_PASSWORD,  # bisa diganti key_file kalau pakai private key
#         remote_bind_address=('127.0.0.1', 3306),
#         local_bind_address=('127.0.0.1', 3307)
#     )
#     server.start()
#     print(f"[INFO] SSH Tunnel aktif di port lokal {server.local_bind_port}")


# def database(app):
#   global db, server
#   if 'db' not in globals():
#     start_ssh_tunnel()
#     app.config['SQLALCHEMY_DATABASE_URI'] = f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@127.0.0.1:{server.local_bind_port}/{MYSQL_DB}"
#     app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
#     db = SQLAlchemy(app)
#     print("[INFO] Database connected")
#   return db

# def generateDatabase(main_app,db):
#   with main_app.app_context():
#     db.create_all()

from flask_sqlalchemy import SQLAlchemy
from src.config.config import DATABASE_URL

def database(app):
  global db
  if 'db' not in globals():
    app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    db = SQLAlchemy(app)
    print("[INFO] Database connected")
  return db
def generateDatabase(main_app,db):
  with main_app.app_context():
    db.create_all()
