# import modules & set up logging
import os
import re
import twokanize
import string
import simplejson as json
from nltk.corpus import stopwords


def any2utf8(text, errors='strict', encoding='utf8'):
    """Convert a string (unicode or bytestring in `encoding`), to bytestring in utf8."""
    if isinstance(text, unicode):
        return text.encode('utf8')
    # do bytestring -> unicode -> utf8 full circle, to ensure valid utf8
    return unicode(text, encoding, errors='ignore').encode('utf8')

def first_pass(text):
    out = text.replace('+', ' _plus_ ').replace(';',' ').replace('-', ' _minus_ ').replace('%', ' _percent_ ').replace('#','').replace('amp', ' ').replace('?',' _qmark_ ').replace('!',' _emark_ ')
    out = re.sub('http\S+',' ', out)
    out = re.sub("\$\d+", " _cash_quantity_ ", out)
    out = re.sub("\$\w+", " _cashtag_ ", out)
    out = re.sub("(\d+)(m)",  r'\1 millions' , out)
    out = re.sub("(\d+)(b)",  r'\1 billions' , out)
    out = re.sub("\@\w+ ", r' _username_ ' , out)
    out = replace_numbers(out)
    stp = stopwords.words('english')
    for w in xrange(len(stp)):
        #print stp[w]
        stp[w] = any2utf8(str(w))

    out = ' '.join([word for word in out.split() if word not in stp])
    return out


def replace_numbers( text):
    numsf = re.findall("\d+\.\d+", text)
    numsi = re.findall("\d+", text)
    nums = numsi + numsf
    for n in nums:
        if float(n) < 10.0:
            text = text.replace(n, ' _onetoten_ ')
        elif float(n) < 20.0:
            text = text.replace(n, ' _ten_ ')
        elif float(n) < 50.0:
            text = text.replace(n, ' _fifty_ ')
        elif float(n) < 100.0:
            text = text.replace(n, ' _hundred_ ')
        else:
            text = text.replace(n, ' _hundredmore_ ' )
    return text


def normalize(sentence):
    text = sentence.lower().replace('\n',' ')
    text = first_pass(text)
    text = twokanize.tokenize(text)
    text = ' '.join(text)
    text = text.translate(None, string.punctuation)
    return text



def normalize_news(sentence):
    text = sentence.lower().replace('\n',' ')
    text = first_pass(text)
    text = twokanize.tokenize(text)
    text = ' '.join(text)
    text = any2utf8(text)
    text = text.translate(None, string.punctuation)
    
    return text


def process_crawled(dirname):
    for fname in os.listdir(dirname):
        count = 0
        newfilepath = os.path.join(dirname, fname + '_tokenized')
        out = open(newfilepath, 'wb')
        for line in open(os.path.join(dirname, fname)):
            count += 1
            print count
            text = line.lower().replace('\n',' ')
            text = first_pass(text)
            text = twokanize.tokenize(text)
            text = ' '.join(text)
            text = text.translate(None, string.punctuation)
            out.write(text+'\n')
        out.close()

def news_crawled(dirname):
    for fname in os.listdir(dirname):
        count = 0
        newfilepath = os.path.join(dirname, fname + '_tokenized')
        out = open(newfilepath, 'wb')
        for line in open(os.path.join(dirname, fname)):
            count += 1

            text = line.lower().replace('\n',' ')
            text = normalize_news(text)
            out.write(text+'\n')
        out.close()



def create_sentences():
    tweets = 'Microblog_Trainingdata-full.json'
    f = open(tweets)
    data = json.load(f)
    out = open('training_sentences','wb')
    for i in xrange(len(data)):
        if data[i]:
            if 'text' in data[i]:
                out.write(data[i]['text'].encode('utf-8').replace('\n','')+'\n')                 
            elif 'message' in data[i]:
                out.write( data[i]['message']['body'].encode('utf-8').replace('\n','')+'\n')                 


if __name__ == '__main__':
    #create_sentences()
    #process_crawled()
    news_crawled()







