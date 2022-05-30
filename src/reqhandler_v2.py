import pandas as pd
import numpy as np

from nltk.corpus import stopwords
from pymystem3 import Mystem
from difflib import SequenceMatcher as SM
from nltk.util import ngrams
import codecs

import demoji

from sklearn.feature_extraction.text import TfidfVectorizer

from src.bert_dataset import CustomDataset
from src.bert_classifier import BertClassifier
import torch
from transformers import logging as transformer_logging

transformer_logging.set_verbosity_error()

import logging
logging.basicConfig(
    level=logging.INFO,
    format=' %(asctime)s - %(message)s',
    datefmt='%H:%M')

printlog = True

def log(msg):
    if printlog:
        logging.info(msg)

import itertools
import string
from fuzzywuzzy import fuzz, process

# https://github.com/asyncee/python-obscene-words-filter
from obscene_words_filter import conf
from obscene_words_filter.words_filter import ObsceneWordsFilter

from sqlalchemy import create_engine

hdir = './'
odir = f'{hdir}obtain/'
sdir = f'{hdir}scrub/'
mdir = f'{hdir}model/'

def preprocess_comments(new_comments, labelled_file_path, tfidf_thresh=0.7):

    f = ObsceneWordsFilter(conf.bad_words_re, conf.good_words_re)

    s_all = ['п еиё з д',
    'х у йёяию',
    'о х у е втл',
    'п и д оеа р',
    'п и д р',
    'её б а нклт',
    'её б н у т',
    'у её б оа нтк',
    'её б л аои',
    'в ы её б',
    'е б еуё т',
    'св ъь еёи б',
    'б л я',
    'г оа в н',
    'м у д а кч',
    'г ао н д о н',
    'г н д о н,'
    'ч м оы',
    'д е р ь м',
    'ш л ю х',
    'з ао л у п',
    'м ао н д',
    'с у ч а р',
    'д ао л б ао её б']

    obs = []
    for i in s_all:
        s = list(itertools.product(*i.split()))
        for j in s:
            obs.append(''.join(j))

    def swear2(phrase0, obs):
        phrase = phrase0.lower()
        for i in obs:
             if fuzz.partial_ratio(i, phrase) > 75:
                    return 1
        return 0

    cols = ['created', 'content', 'rating', 'review_id']
    c3 = new_comments[cols].copy()

    # https://stackoverflow.com/a/68047265/6950776
    mystem = Mystem()
    from string import punctuation
    punctuation0 = ',.;'
    punctuation = punctuation.replace('_', '').replace('-', '')
    for i in punctuation0:
        punctuation = punctuation.replace(i, '')

    russian_stopwords = stopwords.words("russian")
    russian_stopwords.pop(russian_stopwords.index('не'))

    s = '0123456789'
    for i in list(s):
        punctuation += i

    russian_stopwords.append('чо')
    russian_stopwords.append('чтоле')
    russian_stopwords.append('штоле')
    russian_stopwords.append('вж')
    russian_stopwords.append('почему')
    russian_stopwords.append('бк')
    russian_stopwords.append('эх')
    russian_stopwords.append('что')
    russian_stopwords.append('пр')
    russian_stopwords.append('мой')
    russian_stopwords.append('самый')
    russian_stopwords.append('свой')
    russian_stopwords.append('этот')
    russian_stopwords.append('это')
    russian_stopwords.append('за')
    russian_stopwords.append('из')

    '''
    select combo_name, count(cnt) cnt from
    (
    with o as (select id order_id
        from orders
    where created_at > '2022-05-01'
    and src = 0 --mobile
    )
    select combo_name, 1 cnt from dish_orders d
    inner join o on o.order_id = d.order_id
    and combo_name notnull) tmp
    group by combo_name
    order by cnt desc
    '''

    repl_dict = {'еду':'еда',
                'тупить':'лагать',
                'плачу':'платить',
                'заработало':'работать',
                'бургер кинг':'бк',
                'bk':'бк',
                'burger king':'бк',
                'до сих пор':'досихпор',
                'так себе':'таксебе'
    }

    coupons = pd.read_excel(f'{odir}coupons.xlsx')
    coupons = coupons['combo_name'].values.tolist()
    coupons = [i.lower() for i in coupons]

    def find_substring(needle, hay):
    #     https://stackoverflow.com/a/31433394/6950776

        needle_length  = len(needle.split())
        max_sim_val    = 0
        max_sim_string = u""

        for ngram in ngrams(hay.split(), needle_length + int(.2*needle_length)):
            hay_ngram = u" ".join(ngram)
            similarity = SM(None, hay_ngram, needle).ratio()
            if similarity > max_sim_val:
                max_sim_val = similarity
                max_sim_string = hay_ngram

        return max_sim_val, max_sim_string

    def clean_comments(txt, len_thresh=2, verbose=False):

        candidate = demoji.replace(txt.lower(), "")
        (coupon, simil) = process.extractOne(candidate, coupons, scorer=fuzz.token_set_ratio)
        if simil > 88:
            max_sim_val, max_sim_string = find_substring(coupon, candidate)
            candidate = candidate.replace(max_sim_string, 'купон')
        s = candidate
        for i in repl_dict.keys():
            s = s.replace(i, repl_dict[i])
        for i in punctuation:
            if i in s:
                s = s.replace(i, ' ')
        for i in punctuation0:
            if i in s:
                s = s.replace(i, ' ')
        if verbose: print(1,s)
        s = s.replace('  ', ' ')
        s = f.mask_bad_words(s).replace('*', '').split(' ')
        if verbose: print(1,s)
        s = [i for i in s if (i != '')and(i not in russian_stopwords) and (swear2(i, obs)==0)]
        if verbose: print(11,s)
        s = mystem.lemmatize(" ".join(s))
        if verbose: print(1,s)
        s = [i for i in s if not i == ' ']
        s = ' '.join(s)
        if verbose: print(2,s)

        s = s.replace(' - ', '_')
        s = s.replace('не ', 'не_').split(' ')
        if verbose: print(3,s)

        s = [i for i in s if (not len(i) < len_thresh)]
        if verbose: print(4,s)
        if len(s)<2: s = []
        return s

    log('clean and lemmatize comments')
    c3['cleaned_txt'] = c3['content'].apply(lambda x: clean_comments(x))

    t = c3.copy()
    t['comment'] = t['cleaned_txt']
    s = t['comment']

    corpus = []
    for i in s:
        corpus.append(' '.join(i))

    log('compute TF-IDF and clean on threshold')
    # https://towardsdatascience.com/natural-language-processing-feature-engineering-using-tf-idf-e8b9d00e7e76
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names_out()
    dense = vectors.todense()
    denselist = dense.tolist()
    df = pd.DataFrame(denselist, columns=feature_names)

    # be careful with sum
    s = df.sum().reset_index().set_axis(['term', 'freq'], axis=1).sort_values('freq', ascending=False)

    thresh = tfidf_thresh
    shorts = s.query('freq > @thresh').assign(l = lambda row: row['term'].str.len()).\
        sort_values('l').query('l < 4 and term != "смс"')['term'].values.tolist()

    sw = ['что','пр','мой','самый','свой','этот','это','за','из',
        'сразу', 'новый', 'ужас', 'ужасный', 'ужасно', 'пока',
        'хрен', 'пипец', 'конченый','дебильный','дико','днище','быдло',
        'дрянь','фаст','весь', 'нигде','кроме', 'очень', 'тупить', 'тупо',
        'king','ммммм','блин','бомж','безобразие','безобразно','бесить',
        'бесконечно', 'бесконечный', 'бред', 'беспредел', 'еще', 'даун', 'либо'
         ]

    terms = s.query('freq > @thresh')['term'].values.tolist()
    terms = [i for i in terms if (i not in shorts)and(i not in sw)]
    terms2 = terms.copy()

    log('load training file')
    #reading initial terms, which model was trained on
    tmp = pd.read_excel(labelled_file_path)#f'{sdir}t4m_2022_04_20__14_31.xlsx')
    tmp['comment'] = tmp['cleaned_txt'].apply(eval)

    terms = []
    for ind, row in tmp.iterrows():
        for i in row['comment']:
            if not i in terms:
                terms.append(i)

    '''
    s is a set of unique terms. if word in s, we use it.
    if word is in synonims of s, we also use it.
    let's construct dict named di with synonims of s.
    if a word in di, then it has parent word from s in its value
    '''

    log('find synonyms')
    # https://habr.com/ru/post/491448/
    s = terms.copy()
    di = {}
    while terms:
        candidate = terms.pop(0)

        resid = terms2.copy() #residual
        while resid:
            r = resid.pop(0)
            if r == candidate: continue
    #         if fuzz.partial_ratio(r, candidate) > 85:
            if fuzz.token_set_ratio(r, candidate) > 88:
                terms2.pop(terms2.index(r))

                if r in di.keys():
                    di[r] += candidate
                else:
                    di[r] = candidate

    '''
    s is a set of unique terms. if word in s, we use it.
    if word is in synonims of s, we also use it.
    di is a dict with synonims of s.
    if a word in di, then it has parent word from s in its value
    '''
    def comment_clean(lst):
        res = []
        for i in lst:
            if i in s:
                res.append(i)
            elif i in di.keys():
                res.append(di[i])
        return res

    log('leave meaningful words and their synonyms')
    t['comment_clean'] = t['comment'].apply(lambda x: comment_clean(x))

    def clean_short(txt):
        s = demoji.replace(txt.lower(), "")
        for i in repl_dict.keys():
            s = s.replace(i, repl_dict[i])
        for i in punctuation:
            if i in s:
                s = s.replace(i, '')
        for i in punctuation0:
            if i in s:
                s = s.replace(i, ' ')
        return s.split()

    t.drop(columns=['cleaned_txt', 'comment'], inplace=True)
    t.rename(columns={'comment_clean':'cleaned_txt'}, inplace=True)

    log('short preprocessing of short comments')
    t0 = t.copy()
    t0['l'] = t['cleaned_txt'].apply(lambda x: len(x))
    t1 = t0.loc[t0['l']>0].copy()
    t0 = t0.loc[t0['l']==0].copy()
    t0['cleaned_txt'] = t0['content'].apply(lambda x: clean_short(x))
    t = pd.concat([t0, t1], ignore_index=True)
    t['text'] = t['cleaned_txt'].apply(lambda x: ' '.join(x).replace('_', ' ').replace('досихпор', 'до сих пор').replace('таксебе', 'так себе'))

    return t[['review_id', 'text', 'content']]

def assign_class(comments, di, model_name, class_column_name):

    classifier = BertClassifier(
            model_path='cointegrated/rubert-tiny',
            tokenizer_path='cointegrated/rubert-tiny',
            n_classes=len(di),
            epochs=5,
            model_save_path=f'{mdir}bert0.pt'
    )
    classifier.model = torch.load(f'{mdir}{model_name}.pt', map_location=torch.device('cpu'))

    t = comments.copy()

    t[class_column_name] = [classifier.predict(t) for t in t['text'].values.tolist()]
    t[class_column_name] = t[class_column_name].map(di)

    return t


def classify_new_comments(ANALYTICS_DB_URL):

    log('get non-processed negative comments')
    def db_get_con_sqlalchemy(url):
        try:
            cn = create_engine(url, echo=False)
        except Exception as e:
            print(f'Error connecting to app database: {e}')
            return None
        return cn

    ENGINE_ANALYTICS = db_get_con_sqlalchemy(ANALYTICS_DB_URL)

    sql1 = '''
        select
            t1.created::date
            , t1.rating
            , t1."content"
            , t1.review_id
        from appfollow_comments t1
        left join appfollow_comments_classification t on t.review_id = t1.review_id
        where t.class_name1 isnull
        and t1.rating < 4
    '''
    sql2 = '''
        select
            t1.created::date
            , t1.rating
            , t1."content"
            , t1.review_id
        from appfollow_comments t1
        left join appfollow_comments_classification t on t.review_id = t1.review_id
        where t.class_name1 notnull
        and t1.rating < 4
        order by created desc
        limit 2500
    '''

    with ENGINE_ANALYTICS.connect() as conn:
        t1 = pd.read_sql_query(sql1, conn)
        t2 = pd.read_sql_query(sql2, conn)
        conn.close()

    t = pd.concat([t1, t2], ignore_index=False).head(2344)
    ids = t1['review_id'].values.tolist()

    if t1.shape[0] < 1:
        raise Exception('no new negative comments!')

    log('--> preprocess comments')
    # preprocessing
    tmp = preprocess_comments(t, f'{sdir}t4m_2022_04_20__14_31.xlsx').query('review_id in @ids')

    log('--> assign class 1')
    ###########################
    # model 1 to assign class 1
    di = {0: 'uxui',
        1: 'аккаунт',
        2: 'глюки_баги_тормоза',
        3: 'долгое_ожидание_доставки',
        4: 'другое',
        5: 'код_смс',
        6: 'купоны',
        7: 'лояльность',
        8: 'не_возвращаются_деньги_отмененного_заказа',
        9: 'не_работает_доставка',
        10: 'обновление',
        11: 'обслуживание',
        12: 'оплата',
        13: 'создание_заказа'}

    tmp1 = assign_class(tmp, di, model_name='bert', class_column_name='class_name1')

    log('--> preprocess comments')
    # preprocessing
    tmp = preprocess_comments(t, f'{sdir}t4m_2022_05_05__21_59.xlsx').query('review_id in @ids')

    log('--> assign class 2')
    ###########################
    # model 2 to assign class 2
    di = {0: 'uxui',
        1: 'аккаунт',
        2: 'глюки_баги_тормоза',
        3: 'долгое_ожидание_доставки',
        4: 'доставка_общее',
        5: 'другое',
        6: 'купоны',
        7: 'лояльность',
        8: 'не_возвращаются_деньги_отмененного_заказа',
        9: 'обновление',
        10: 'обслуживание',
        11: 'оплата',
        12: 'регистрация/коды',
        13: 'создание_заказа',
        14: 'цена'}

    tmp2 = assign_class(tmp, di, model_name='bert2', class_column_name='class_name2')

    res = tmp1[['review_id', 'class_name1']].merge(tmp2[['review_id', 'class_name2']], how='inner', on='review_id')

    log('--> append to database')

    with ENGINE_ANALYTICS.connect() as conn:
         res.to_sql('appfollow_comments_classification', conn, if_exists='append', index=False)
         conn.close()

    print(f'{res.shape[0]} lines added')

    return res
