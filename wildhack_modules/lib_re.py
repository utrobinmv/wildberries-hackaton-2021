import re

def preprocessed_text(row, name, name_sec, manufacture):
    cell = row[name] + ' ' + row[name_sec]
    manufacture_cell = row[manufacture]
    if manufacture_cell not in cell:
        cell = cell + ' ' + manufacture_cell
    separate_cell = re.sub(r'[0-9][А-Яа-яA-Za-z]|[А-Яа-яA-Za-z][0-9]', lambda m: ' '.join(m.group()), cell)
    clear_cell = re.sub(r'[-\.,)(_/\"\'+*xXхХ]', ' ', separate_cell)
    first_final_cell = clear_cell.lower().split()
    second_final_cell = re.sub(r'[-\.,)(_/\"\'+*xXхХ]', '', cell).lower().split()
    final_cell = first_final_cell + list(set(second_final_cell) - set(first_final_cell))
    return final_cell

def tokenize_text_simple_regex(txt, min_token_size=4, not_lower = False):
    if not_lower == False:
        txt = txt.lower()
    TOKEN_RE = re.compile(r'(?u)\d+|[a-zA-Zа-яА-Я]+')
    all_tokens = TOKEN_RE.findall(txt)
    return [token for token in all_tokens if len(token) >= min_token_size]

def tokenize_corpus(texts, tokenizer=tokenize_text_simple_regex, **tokenizer_kwargs):
    #print(tokenizer_kwargs)
    return [tokenizer(text, **tokenizer_kwargs) for text in texts]