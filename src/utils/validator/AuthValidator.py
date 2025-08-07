
from pydantic import BaseModel, EmailStr, constr

class RegisterValidator(BaseModel):
  email: EmailStr
  password: constr(min_length=8, max_length=16)
  name: str
  password: str

class LoginValidator(BaseModel):
  email: EmailStr
  password: constr(min_length=8, max_length=16)
