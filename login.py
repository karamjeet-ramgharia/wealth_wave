"""Module to get request token """
import time
import logging
from selenium import webdriver
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.common.exceptions import SessionNotCreatedException
import pyotp

logging.getLogger("requests").setLevel(logging.WARNING)


def kiteLogin(user_id, user_pwd, totp_key, api_key):
    """Function to get request token """
    max_tries = 20
    count = 0
    driver = None
    try:
        service = Service()
        options = webdriver.ChromeOptions()
        driver = webdriver.Chrome(service=service, options=options)
        # driver = webdriver.Chrome(
        #     r'C:\Users\HP\Documents\CHROME DRIVER\chromedriver.exe')
        driver.get(f'https://kite.trade/connect/login?api_key={api_key}&v=3')
    except SessionNotCreatedException as e:
        # print(e, type(e))
        print(e.msg, type(e.msg))

    assert isinstance(
        driver, webdriver.Chrome), "Please contact admin regarding the issue"

    login_id = WebDriverWait(driver, 1).until(
        lambda x: x.find_element(by=By.XPATH, value='//*[@id="userid"]'))

    login_id.send_keys(user_id)
    pwd = WebDriverWait(driver, 1).until(
        lambda x: x.find_element(by=By.XPATH, value='//*[@id="password"]'))
    # lambda x: x.find_element_by_xpath('//*[@id="password"]'))
    pwd.send_keys(user_pwd)

    submit = WebDriverWait(driver, 1).until(lambda x: x.find_element(
        by=By.XPATH, value='//*[@id="container"]/div/div/div[2]/form/div[4]/button'))
    submit.click()

    time.sleep(1)
    totp = WebDriverWait(driver, 5).until(
        lambda x: x.find_element(by=By.XPATH, value='//*[@icon="shield"]'))
    authkey = pyotp.TOTP(totp_key)
    toSendAuthKey = authkey.now()
    totp.send_keys(toSendAuthKey)

    url = ""
    while count < max_tries:
        time.sleep(0.5)
        url = driver.current_url
        if "request_token=" in url:
            break
        count = count + 1
    initial_token = url.split('request_token=')[1]
    request_token = initial_token.split('&')[0]

    driver.close()

    # kite = KiteConnect(api_key = api_key)
    # print(request_token)
    # data = kite.generate_session(request_token, api_secret=api_secret)
    # kite.set_access_token(data['access_token'])
    # print("request_token  ",request_token)
    return request_token


if __name__ == "__main__":
    pass