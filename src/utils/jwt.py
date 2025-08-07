import jwt
from src.config import config
def encode(data:dict):
  payload= {
    "user_id": data['user_id'],
    "name": data['name'],
    "email": data['email'],
    
  }
  return jwt.encode(payload, config.JWT_ACCESS_TOKEN_SECRET,config.JWT_ACCESS_TOKEN_ALGORITHM)
  
def decode(token):
  return jwt.decode(token, config.JWT_ACCESS_TOKEN_SECRET,config.JWT_ACCESS_TOKEN_ALGORITHM)