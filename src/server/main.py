from flask import Flask 
from flask_cors import CORS
from src.config.config import BASE_URL, PORT, DEBUG
from flask_migrate import Migrate
from src.config.database import database

main_app = Flask(
    __name__,
    static_folder="../../public",
    static_url_path="/",
)
CORS(main_app)
db = database(main_app)
migrate = Migrate(main_app, db)


proses_topic_association = db.Table('proses_topic_association',
    db.Column('Proses_id', db.Integer, db.ForeignKey('steam_id_proses.Proses_id'), primary_key=True),
    db.Column('Topic_id', db.Integer, db.ForeignKey('topic_modeling.Topic_id'), primary_key=True)
)


print("Server is running on url: " + BASE_URL + ":", PORT)

if __name__ == "__main__":
    main_app.run(debug=DEBUG, host=BASE_URL, port=PORT)
