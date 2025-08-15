from dotenv import dotenv_values

env = dotenv_values(".env")
BASE_URL = env["BASE_URL"] or "http://localhost"
PORT = env["PORT"] or 5000
DEBUG = env["DEBUG"] or True
GROQ_API_KEY = env["GROQ_API_KEY"] or "your_groq_api_key_here"
ARCHETYPE_API_URL = env["ARCHETYPE_API_URL"] or "http://localhost:5000"

DATABASE_URL = env["DATABASE_URL"] or "mysql://root:root@localhost:3306/payer-segmentasi"

JWT_ACCESS_TOKEN_EXPIRES = env["JWT_ACCESS_TOKEN_EXPIRES"] or 60 * 60 * 24 * 7
JWT_ACCESS_TOKEN_SECRET = env["JWT_ACCESS_TOKEN_SECRET"] or "secret"
JWT_ACCESS_TOKEN_ALGORITHM = env["JWT_ACCESS_TOKEN_ALGORITHM"] or "HS256"

# SSH Tunnel Config
SSH_HOST = "104.214.168.165"
SSH_PORT = 22
SSH_USERNAME = "nuruldinikamilah"
SSH_PASSWORD = "Nurulsnaily123"


# MySQL Config
MYSQL_USER = "root"
MYSQL_PASSWORD = "password"
MYSQL_DB = "player_segmentasi"

