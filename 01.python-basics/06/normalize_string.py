# 문자열 데이터를 분석하기 전 처리함수 만들기
# 1. 공백제거
# 2. 필요 없는 문장 부호 제거
# 3. 대소문자 정리(Capitalize 처리)

import re

states = ['Alabama', ' Georgia!', 'Georgia ', 'georgia', 'FlOrIda', 'south carolina   ', 'West virginia?']


def clean_strings(strings):
        results = []
        for string in strings:
            string = string.strip()
            string = re.sub('[!#?]', '', string)
            string = string.title()
            results.append(string)

        return results


states = clean_strings(states)
print(states)


###########################################

states = ['Alabama', ' Georgia!', 'Georgia ', 'georgia', 'FlOrIda', 'south carolina   ', 'West virginia?']


def clean_strings(strings, *funcs):
    results = []

    for string in strings:
        for func in funcs:
            string = func(string)
        results.append(string)
    return results

# def remove_specialcharactor(string):
#    return re.sub('[!#?]', '', string)


states = clean_strings(states, str.strip, str.title, lambda s: re.sub('[!#?]', '', s))
print(states)

