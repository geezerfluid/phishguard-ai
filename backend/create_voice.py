from gtts import gTTS

text = "Your order has been delivered successfully. Thank you for shopping with us."

tts = gTTS(text=text, lang='en')
tts.save("real_voice.mp3")

print("Phishing voice created!")