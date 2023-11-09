import telebot
import os
from dotenv import load_dotenv
import tele_model as model

load_dotenv()

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

bot = telebot.TeleBot(TOKEN)
bot.set_webhook()

@bot.message_handler(commands=['start'])
def start(message):
    """
    Bot will introduce itself upon /start command, and prompt user for his request
    """
    try:
        # Start bot introduction
        start_message = "Hello! Ask me anything about migrant workers' medical, dental and mental health converage in Singapore! \n\n হ্যালো! সিঙ্গাপুরে অভিবাসী শ্রমিকদের চিকিৎসা, ডেন্টাল এবং মানসিক স্বাস্থ্য সম্পর্কে কিছু জিজ্ঞাসা করুন! \n\n 你好！请向我询问有关新加坡外籍劳工的医疗、牙科和心理健康状况的任何问题！\n\n வணக்கம்! சிங்கப்பூரில் புலம்பெயர்ந்த தொழிலாளர்களின் மருத்துவம், பல் மருத்துவம் மற்றும் மனநலம் பற்றி என்னிடம் ஏதாவது கேளுங்கள்!"
        bot.send_message(message.chat.id, start_message)

    except Exception as e:
        bot.send_message(
            message.chat.id, 'Sorry, something seems to gone wrong! Please try again later!')


@bot.message_handler(content_types=['text'])
def send_text(message):
    response = model.getResponse(message.text)
    bot.send_message(message.chat.id, response)

def main():
    """Runs the Telegram Bot"""
    print('Loading configuration...') # Perhaps an idea on what you may want to change (optional)
    print('Successfully loaded! Starting bot...')
    bot.infinity_polling()


if __name__ == '__main__':
    main()
