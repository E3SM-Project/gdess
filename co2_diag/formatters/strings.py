import os, re


def tex_escape(text):
    """Return text with problematic escape sequences parsed for Latex use.

    Note: This function was copied from the following StackOverflow answer,
        <https://stackoverflow.com/a/25875504/10134974>

    Parameters
    ----------
    text
        a plain text message

    Returns
    -------
    the message escaped to appear correctly in LaTeX
    """
    conv = {
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '_': r'\_',
        '{': r'\{',
        '}': r'\}',
        '~': r'\textasciitilde{}',
        '^': r'\^{}',
        '\\': r'\textbackslash{}',
        '<': r'\textless{}',
        '>': r'\textgreater{}',
    }
    regex = re.compile('|'.join(re.escape(str(key))
                                for key
                                in sorted(conv.keys(), key=lambda item: - len(item))))
    return regex.sub(lambda match: conv[match.group()], text)


def append_before_extension(filename: str, text_to_append: str) -> str:
    """Take a filename and add a string to the end before the extension.
    """
    return "{0}_{2}{1}".format(*os.path.splitext(filename) + (text_to_append,))