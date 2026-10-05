"""Заготовки задач на базовый Python."""

from grader_contracts.python_basics import PositiveIntegerInput, TextInput, VectorPairInput


def count_vowels(data: TextInput) -> int:
    text = data.value
    vowels = set('aeiouAEIOU')
    return sum(1 for char in text if char in vowels)


def has_unique_characters(data: TextInput) -> bool:
    text = data.value
    return len(set(text)) == len(text)


def count_one_bits(data: PositiveIntegerInput) -> int:
    number = data.value
    return bin(number).count('1')


def multiplicative_persistence(data: PositiveIntegerInput) -> int:
    number = data.value
    count = 0
    while number >= 10:
        product = 1
        for digit in str(number):
            product *= int(digit)
        number = product
        count += 1
    return count


def mse(data: VectorPairInput) -> float:
    predicted, expected = data.predicted, data.expected
    n = len(predicted)
    return sum((p - e) ** 2 for p, e in zip(predicted, expected)) / n


def prime_factorization(data: PositiveIntegerInput) -> str:
    number = data.value
    factors = {}
    d = 2
    while d * d <= number:
        while (number % d) == 0:
            factors[d] = factors.get(d, 0) + 1
            number //= d
        d += 1
    if number > 1:
        factors[number] = factors.get(number, 0) + 1
    
    result = []
    for prime in sorted(factors.keys()):
        power = factors[prime]
        if power == 1:
            result.append(f"({prime})")
        else:
            result.append(f"({prime}**{power})")
    return "".join(result)


def pyramid(data: PositiveIntegerInput) -> int | str:
    cube_count = data.value
    k = 1
    current_sum = 0
    while current_sum < cube_count:
        current_sum += k ** 2
        if current_sum == cube_count:
            return k
        k += 1
    return "It is impossible"


def is_balanced_number(data: PositiveIntegerInput) -> bool:
    number = data.value
    s = str(number)
    length = len(s)
    mid = length // 2
    if length % 2 == 0:
        left = s[:mid]
        right = s[mid:]
    else:
        left = s[:mid]
        right = s[mid+1:]
    
    sum_left = sum(int(digit) for digit in left)
    sum_right = sum(int(digit) for digit in right)
    return sum_left == sum_right
