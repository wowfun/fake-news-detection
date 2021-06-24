import pickle
import tensorflow as tf


# 词向量化
def tokenize(lang, mode='load', path=None, max_num_words=None, max_sequence_len=256):  # mode: create or load
    if mode == 'load':
        with open(path, 'rb') as handle:
            lang_tokenizer = pickle.load(handle)
        print('** Load tokenzier from: ', path)
    else:
        lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=max_num_words,
                                                               filters='"#$%&()*+,-./:;<=>@[\\]^_`{|}~', lower=True)
        lang_tokenizer.fit_on_texts(lang)
        # saving
        with open(path, 'wb') as handle:
            pickle.dump(lang_tokenizer, handle,
                        protocol=pickle.HIGHEST_PROTOCOL)
        print('** Save tokenizer at: ', path)

    tensor = lang_tokenizer.texts_to_sequences(lang)

    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, maxlen=max_sequence_len,
                                                           padding='post', truncating='post')  # NOTE
    print('** Total different words: %s.' % len(lang_tokenizer.word_index))

    return tensor, lang_tokenizer