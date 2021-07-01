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
