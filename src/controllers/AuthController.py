from flask import Blueprint, g, request
from src.middlewares.AuthMiddleware import isAuthenticated
from src.services.AuthService import AuthService as AuthService
import src.utils.getResponse as Response

AuthApp = Blueprint("AuthApp", __name__)
authService = AuthService()


@AuthApp.route("/login", methods=["GET"])
def index():
    users = []
    return Response.success(users, "success get all user")


@AuthApp.route("/register", methods=["POST"])
def register():
    req = request.json
    result = authService.registerUser(req)
    if result["status"] == "failed":
        return Response.error(result["data"], result["code"])
    return Response.success(result["data"], "success create new user")


@AuthApp.route("/login", methods=["POST"])
def login():
    req = request.json
    result = authService.login(req)
    if result["status"] == "failed":
        return Response.error(result["data"], result["code"])
    return Response.success(result["data"], "success login")


@AuthApp.route("/me", methods=["GET"])
@isAuthenticated
def me():
    return Response.success(g.user, "success get user data")


@AuthApp.route("/verify", methods=["post"])
@isAuthenticated
def verify():
    req = request.json

    result = authService.verify(req)
    if result["status"] == "failed":
        return Response.error(result["data"], result["code"])
    return Response.success(result["data"], "success verify user")
