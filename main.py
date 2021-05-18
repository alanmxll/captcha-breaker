from captcha_resolver import break_captcha

if __name__ == "__main__":
    captcha_text = break_captcha()
    print(f'CAPTCHA text: {captcha_text}')
