import re
from functools import partial, lru_cache


@lru_cache()
def minify_css(css: str,
               _whitespace=partial(re.compile(r'\s*([,{:;}])\s*').sub, r'\1'),
               _comments=partial(re.compile(r'/\*.*?\*/', flags=re.DOTALL).sub, ''),
               _trailing_semicolon=partial(re.compile(r';\s*}').sub, '}')):
    """
    Minify CSS by removing extraneous whitespace, comments, and trailing semicolons.
    """
    return _trailing_semicolon(_whitespace(_comments(css))).strip()