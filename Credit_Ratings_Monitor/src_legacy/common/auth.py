import time
from datetime import datetime
from typing import Optional

import jwt
import pytz
import requests
import streamlit as st

from common.cookie import CookieHandler

CLERK_URL = "https://clerk.bigdata.com"
BROWSER = "client"
CLERK_JS_VERSION = "4.70.5"

CLERK_API_URL = "https://api.clerk.com/v1/"

SESSION_EXPIRATION = 60 * 15

BIGDATA_LOGO = "https://bigdata.com/assets/svg/bigdata-logo-white.svg"
WIDE_LAYOUT = True


class Auth:

    def __init__(self, url: str, browser: str, clerk_js_version: str):
        self.url = url
        self.browser = browser
        self.clerk_js_version = clerk_js_version
        self.session_id = None
        self.session_data = None
        self.organization_id = None
        self.token = None

    def is_authenticated(self) -> bool:
        return self.token is not None

    def authenticate(self, username: str, password: str) -> bool:
        self.token = self.get_clerk_token(username, password)
        return self.is_authenticated()

    def get_clerk_token(self, username: str, password: str) -> Optional[str]:
        try:
            dev_session = self._get_dev_session()
            self.session_data, cookies = self._sign_in(dev_session, username,
                                                       password)
            self.session_id = self._extract_session_id()
            self.organization_id = self._extract_organization_id()
            self._touch_session(dev_session, cookies)
            self.token = self._get_jwt_token(dev_session, cookies)
            return self.token
        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return None

    def get_token_claim(self, token: str, claim: str) -> Optional[str]:
        try:
            return jwt.decode(token, options={
                "verify_signature": False
            }).get(claim)
        except Exception as e:
            print(f"Error decoding token: {e}")
            return None

    def _get_dev_session(self) -> str:
        request_url = f"{CLERK_URL}/v1/{BROWSER}?_clerk_js_version={CLERK_JS_VERSION}"
        headers = {"Content-Type": "application/json"}
        response = requests.post(request_url, headers=headers)
        response.raise_for_status()
        return response.json().get("token")

    def _sign_in(self, dev_session: str, username: str, password: str):
        request_url = f"{CLERK_URL}/v1/client/sign_ins?_clerk_js_version={CLERK_JS_VERSION}&__dev_session={dev_session}"
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        data = {"identifier": username, "password": password}
        response = requests.post(request_url, headers=headers, data=data)
        response.raise_for_status()
        return response.json(), response.cookies

    def _extract_session_id(self) -> str:
        return self.session_data.get("client", {}).get("sessions",
                                                       [{}])[0].get("id")

    def get_user_data(self) -> dict:
        return (self.session_data.get("client",
                                      {}).get("sessions",
                                              [{}])[0].get("public_user_data"))

    def _extract_organization_id(self) -> Optional[str]:
        return (self.session_data.get("client", {}).get(
            "sessions", [{}])[0].get("last_active_organization_id"))

    def _touch_session(self, dev_session: str, cookies):
        request_url = f"{CLERK_URL}/v1/client/sessions/{self.session_id}/touch?_clerk_js_version={CLERK_JS_VERSION}&__dev_session={dev_session}"
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        data = {"active_organization_id": self.organization_id}
        response = requests.post(request_url,
                                 headers=headers,
                                 data=data,
                                 cookies=cookies)
        response.raise_for_status()

    def _get_jwt_token(self, dev_session: str, cookies) -> str:
        request_url = f"{CLERK_URL}/v1/client/sessions/{self.session_id}/tokens/qa_template_jnbkds?_clerk_js_version={CLERK_JS_VERSION}&__dev_session={dev_session}"
        headers = {"Content-Type": "application/json"}
        response = requests.post(request_url, headers=headers, cookies=cookies)
        response.raise_for_status()
        return response.json().get("jwt")


class Authenticator:

    def __init__(self):
        self.authentication_handler = Auth(CLERK_URL, BROWSER,
                                           CLERK_JS_VERSION)
        self.cookie_handler = CookieHandler(cookie_name="bigdata_cookie",
                                            cookie_key="bigdata_key")

        if "username" not in st.session_state:
            st.session_state["username"] = None
        if "authentication_status" not in st.session_state:
            st.session_state["authentication_status"] = False
        if "user_data" not in st.session_state:
            st.session_state["user_data"] = None

    def login(
            self,
            location: str = "main",
            fields: dict = {},
            clear_on_submit: bool = False,
    ) -> tuple:
        """
        Creates a login widget.

        Parameters
        ----------
        location: str
            Location of the login widget i.e. main or sidebar.
        max_concurrent_users: int
            Maximum number of users allowed to login concurrently.
        max_login_attempts: int
            Maximum number of failed login attempts a user can make.
        fields: dict
            Rendered names of the fields/buttons.
        clear_on_submit: bool
            Clear on submit setting, True: clears inputs on submit, False: keeps inputs on submit.

        Returns
        -------
        str
            Name of the authenticated user.
        bool
            Status of authentication, None: no credentials entered,
            False: incorrect credentials, True: correct credentials.
        str
            Username of the authenticated user.
        """
        if not fields:
            fields = {
                "Form name": "Login",
                "Username": "Username",
                "Password": "Password",
                "Login": "Login",
            }

        if not st.session_state.get("authentication_status"):
            token = self.cookie_handler.get_cookie()
            if token:
                if token and token["exp_date"] > datetime.now(
                        pytz.UTC).timestamp():
                    st.session_state["username"] = token["username"]
                    st.session_state["user_data"] = token["user_data"]
                    st.session_state["authentication_status"] = True
                # self.authentication_handler.execute_login(token=token)
            time.sleep(0.7)
            if not st.session_state.get("authentication_status"):
                if location == "main":
                    login_container = get_login_container(
                        wide_layout=WIDE_LAYOUT)
                    login_form = login_container.form(
                        "Login", clear_on_submit=clear_on_submit)
                elif location == "sidebar":
                    login_form = st.sidebar.form("Login")
                login_form.subheader("Login" if "Form name" not in
                                                fields else fields[
                    "Form name"])
                username = login_form.text_input(
                    "Username" if "Username" not in
                                  fields else fields["Username"]).lower()
                password = login_form.text_input(
                    "Password"
                    if "Password" not in fields else fields["Password"],
                    type="password",
                )
                login_form.info("Enter your Bigdata credentials")
                if login_form.form_submit_button("Login" if "Login" not in
                                                            fields else fields[
                    "Login"]):
                    if self.authentication_handler.get_clerk_token(
                            username, password):
                        st.session_state["username"] = username
                        st.session_state["user_data"] = (
                            self.authentication_handler.get_user_data())
                        st.session_state["authentication_status"] = True
                        st.session_state["logout"] = False
                        self.cookie_handler.set_cookie()
        return (
            st.session_state["username"],
            st.session_state["authentication_status"],
            st.session_state["user_data"],
        )

    def logout(self):
        self.cookie_handler.delete_cookie()
        st.session_state.clear()
        st.session_state["username"] = None
        st.session_state["user_data"] = None
        st.session_state["authentication_status"] = None
        st.session_state["logout"] = True
        time.sleep(0.7)


def bigdata_session_init():
    auth = Authenticator()
    if not st.session_state.get("authentication_status"):
        auth_result = auth.login()
        if not auth_result[1]:
            st.stop()

    if st.session_state.get("logout"):
        auth.logout()
        st.stop()


def log_out():
    st.session_state["logout"] = True
    st.session_state["authentication_status"] = False


def get_login_container(wide_layout=True, include_logo=True):
    if wide_layout:
        _, login_container, _ = st.columns([2, 3, 2])
    else:
        login_container = st.container()
    if include_logo:
        _, col, _ = login_container.columns(3)
        logo_container = col.container()
        logo_container.image(BIGDATA_LOGO, use_column_width=True)
    return login_container
