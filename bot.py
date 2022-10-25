#telegram import
import logging
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext

#sklearn import
import nltk

# nltk.download('vader_lexicon')

import textstat
import docx2txt
import pandas as pd
import numpy as np
from collections import Counter
import math
import string
from docx import Document
import glob
import os
import textstat
import language_tool_python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
import matplotlib.pyplot as plt
from nltk.sentiment import SentimentIntensityAnalyzer

PORT = int(os.environ.get('PORT', 5000))
#sklearn code
df = pd.read_csv('learningData/dataset.csv')
df = pd.DataFrame(df)
X = np.asarray(df[df.columns.difference(['Essay', 'Score_band'])])
Y = np.asarray(df['Score_band'])

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.01, shuffle=False)

reg = linear_model.Ridge (alpha = 1)
reg.fit(np.nan_to_num(X_train), np.nan_to_num(y_train))

def counter_words(essay):
    tokens = nltk.word_tokenize(essay.lower())
    text = nltk.Text(tokens)
    tags = nltk.pos_tag(text)
    counts = Counter(tag for word,tag in tags if word not in string.punctuation)
    return counts

def sentiment(x):
    sia = SentimentIntensityAnalyzer()
    return sia.polarity_scores(x)

def testessaystat(test_data):
    flesch = textstat.flesch_reading_ease(test_data)
    smog = textstat.smog_index(test_data)
    flesch_kin = textstat.flesch_kincaid_grade(test_data)
    coleman = textstat.coleman_liau_index(test_data)
    ari = textstat.automated_readability_index(test_data)
    dcrs = textstat.dale_chall_readability_score(test_data)
    dw = textstat.difficult_words(test_data)
    lwf = textstat.linsear_write_formula(test_data)
    gf = textstat.gunning_fog(test_data)
    ts = textstat.text_standard(test_data, float_output = True)
    return {'flesch':flesch, 'smog': smog, 'coleman': coleman, 'ari': ari, 'dcrs': dcrs, 'dw': dw, 'lwf': lwf, 'gf': gf, 'ts':ts }


def check_grammar(text):
    tool = language_tool_python.LanguageToolPublicAPI('en-US')
    matches = tool.check(text)
    return {'N mistakes':len(matches)}


def checker(text):
    tes = testessaystat(text)
    chg = check_grammar(text)
    st = sentiment(text)
    cw = counter_words(text)
    test_essay = {**tes, **chg, **st, **cw}
    for v in df.columns:
        if v not in test_essay and v not in ['Essay', 'Score_band']:
            test_essay[v] = 0
    for n in test_essay:
        if v not in df.columns:
            test_essay.pop('n')
    sorted_dict = {}
    for i in list(df.columns.difference(['Essay', 'Score_band'])):
        for k in list(test_essay.keys()):
            if k == i:
                sorted_dict[k] = test_essay[k]
                break
    essay_t = [*sorted_dict.values()]
    result = reg.predict([essay_t])
    if result[0] >= 8.5:
        result = 8.5
    else:
        result = (math.floor(result[0]))
    if len(text) < 230:
        return('Your essay is too short and your band for it will not be more than 4.5')
    return (f'Your result is: {result}, accuracy is +-0.5. Take into account that topic achievement is not marked!!!')





# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)

logger = logging.getLogger(__name__)


# Define a few command handlers. These usually take the two arguments update and
# context. Error handlers also receive the raised TelegramError object in error.
def start(update, context: CallbackContext) -> None:
    """Send a message when the command /start is issued."""
    update.message.reply_text('Hi! Please send us your ielts writing 2!')


def help_command(update, context: CallbackContext) -> None:
    """Send a message when the command /help is issued."""
    update.message.reply_text('We are developing help function of this bot. Sorry!')


def echo(update, context: CallbackContext) -> None:
    """Echo the user message."""
    update.message.reply_text('Please, wait for 10 seconds!')
    reply = checker(update.message.text)
    update.message.reply_text(reply)

def error(update, context):
    """Log Errors caused by Updates."""
    logger.warning('Update "%s" caused error "%s"', update, context.error)

TOKEN = '1538265942:AAEWUMMAEKd80fcggKWTxryMESEnEwMzmWU'

def main():
    """Start the bot."""
    # Create the Updater and pass it your bot's token.
    updater = Updater(TOKEN)

    # Get the dispatcher to register handlers
    dispatcher = updater.dispatcher

    # on different commands - answer in Telegram
    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(CommandHandler("help", help_command))

    # on noncommand i.e message - echo the message on Telegram
    dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, echo))

    # Start the Bot
    # updater.start_polling()
    updater.start_webhook(listen="0.0.0.0",
                          port=int(PORT),
                          url_path=TOKEN)

    updater.bot.setWebhook(f'https://essay4scoring-rlhydzw31-akhmedov-i.vercel.app/{TOKEN}')


    # Run the bot until you press Ctrl-C or the process receives SIGINT,
    # SIGTERM or SIGABRT. This should be used most of the time, since
    # start_polling() is non-blocking and will stop the bot gracefully.
    updater.idle()


if __name__ == '__main__':
    main()
