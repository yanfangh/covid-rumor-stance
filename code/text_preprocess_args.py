class dotdict(dict):
  def __getattr__(self, name):
      return self[name]

text_preprocess_args=dotdict({'replace_usernames': True,
              'replace_urls': True,
              'asciify_emojis': True,
              'replace_multiple_usernames': True,
              'replace_multiple_urls': True,
              'remove_unicode_symbols': True,
              'remove_accented_characters': False,
              'standardize_punctuation': True,
              'username_filler': 'twitteruser',
              'url_filler': 'twitterurl'
             })
