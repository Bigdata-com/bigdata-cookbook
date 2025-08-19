from datetime import datetime, timedelta

import extra_streamlit_components as stx
import jwt
import pytz
import streamlit as st
from jwt import DecodeError, InvalidSignatureError


class CookieHandler:
    """
    This class will execute all actions related to the re-authentication cookie,
    including retrieving, deleting, and setting the cookie.
    """

    def __init__(self,
                 cookie_name: str,
                 cookie_key: str,
                 cookie_expiry_days: float = 30.0):
        """
        Create a new instance of "CookieHandler".

        Parameters
        ----------
        cookie_name: str
            Name of the cookie stored on the client's browser for password-less re-authentication.
        cookie_key: str
            Key to be used to hash the signature of the re-authentication cookie.
        cookie_expiry_days: float
            Number of days before the re-authentication cookie automatically expires on the client's
            browser.
        """
        self.cookie_name = cookie_name
        self.cookie_key = cookie_key
        self.cookie_expiry_days = cookie_expiry_days
        self.cookie_manager = stx.CookieManager()
        self.token: str = ""
        self.exp_date = ""

    def get_cookie(self) -> dict:
        """
        Retrieves, checks, and then returns the re-authentication cookie.

        Returns
        -------
        str
            re-authentication cookie.
        """
        self.token = self.cookie_manager.get(self.cookie_name)
        if self.token is not None:
            cookie = self._token_decode()
            if (cookie and ("username" in cookie) and
                (cookie["exp_date"] > datetime.now(pytz.UTC).timestamp())):
                return cookie
        return {}

    def delete_cookie(self):
        """
        Deletes the re-authentication cookie.
        """
        try:
            self.cookie_manager.delete(self.cookie_name)
        except KeyError as e:
            print(e)

    def set_cookie(self):
        """
        Sets the re-authentication cookie.
        """
        self.exp_date = self._set_exp_date()
        token = self._token_encode()
        self.cookie_manager.set(
            self.cookie_name,
            token,
            expires_at=datetime.now() +
            timedelta(days=self.cookie_expiry_days),
        )

    def _set_exp_date(self) -> float:
        """
        Sets the re-authentication cookie's expiry date.

        Returns
        -------
        str
            re-authentication cookie's expiry timestamp in Unix Epoch.
        """
        return (datetime.now(pytz.UTC) +
                timedelta(days=self.cookie_expiry_days)).timestamp()

    def _token_decode(self) -> dict:
        """
        Decodes the contents of the re-authentication cookie.

        Returns
        -------
        str
            Decoded cookie used for password-less re-authentication.
        """
        try:
            return jwt.decode(self.token,
                              self.cookie_key,
                              algorithms=["HS256"])
        except InvalidSignatureError as e:
            print(e)
        except DecodeError as e:
            print(e)
        return {}

    def _token_encode(self) -> str:
        """
        Encodes the contents of the re-authentication cookie.

        Returns
        -------
        str
            Cookie used for password-less re-authentication.
        """
        return jwt.encode(
            {
                "username": st.session_state["username"],
                "user_data": st.session_state["user_data"],  #
                "exp_date": self.exp_date,
            },
            self.cookie_key,
            algorithm="HS256",
        )
