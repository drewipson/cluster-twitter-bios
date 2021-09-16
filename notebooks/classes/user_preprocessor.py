import string, itertools as it, re
from nltk.tokenize.casual import TweetTokenizer
from nltk.corpus import stopwords

class UserPreProcessor():
    def make_punc_stopwords(self, max_length=4):
        """Generates punctuation 'words' up to ``max_length`` characters."""
        def punct_maker(length):
            return ((''.join(x) for x in it.product(string.punctuation,
                                                    repeat=length)))
        words = it.chain.from_iterable((punct_maker(length)
                                        for length in range(max_length+1)))
        return list(words)
    
    def generate_stopwords(self):
        """ Generates a list of various stopwords and punctuation patterns. """
        stopwords = self.make_punc_stopwords()
        for _ in string.punctuation:
            stopwords.append(_)
        return stopwords

    def replace_www(self, description_text: str, replace=None):
        """ Replace internet related strings in bio descriptions. """
        replace = '<-URL->' if replace is None else replace
        pattern = re.compile('(https?://)?(\w*[.]\w+)+([/?=&]+\w+)*')
        return re.sub(pattern, replace, description_text)

    def tweet_tokenizer(self, description_text: str):
        """ Converts description text to a list of token from NLTK and TweetTokenizer. """
        tokenizer = TweetTokenizer(preserve_case = False, reduce_len = True, strip_handles = False)
        tokens = tokenizer.tokenize(description_text)
        return tokens