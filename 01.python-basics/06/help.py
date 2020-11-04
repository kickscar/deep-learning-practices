# help 함수

# help(print)


def plus(a, b):
    """
        return the sun of parameter a, b
        :param a: 1st parameter
        :param b: 1st parameter
        :return: sum
    """
    return a + b


# plus.__doc__ = """
#    return the sun of parameter a, b
#    :param a: 1st parameter
#    :param b: 1st parameter
#    :return: sum
# """


help(plus)


# 참고
print(plus.__doc__)
print(plus.__name__)