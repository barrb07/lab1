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


# ==========================================
# СОБСТВЕННЫЕ ТЕСТЫ
# ==========================================
if __name__ == "__main__":
    # Простые моки для тестирования
    class MockTextInput:
        def __init__(self, value):
            self.value = value
    
    class MockPositiveIntegerInput:
        def __init__(self, value):
            self.value = value
    
    class MockVectorPairInput:
        def __init__(self, predicted, expected):
            self.predicted = predicted
            self.expected = expected
    
    # Задача 1: Подсчёт гласных
    print("Тесты для count_vowels:")
    assert count_vowels(MockTextInput("Hello")) == 2, "Hello -> 2 (e, o)"
    assert count_vowels(MockTextInput("Rhythm")) == 0, "Rhythm -> 0"
    assert count_vowels(MockTextInput("AEIOU")) == 5, "AEIOU -> 5"
    assert count_vowels(MockTextInput("")) == 0, "Пустая строка -> 0"
    assert count_vowels(MockTextInput("aAeEiIoOuU")) == 10, "Все гласные -> 10"
    print("✅ Все тесты пройдены\n")
    
    # Задача 2: Уникальные символы
    print("Тесты для has_unique_characters:")
    assert has_unique_characters(MockTextInput("abc")) == True, "abc -> True"
    assert has_unique_characters(MockTextInput("aabc")) == False, "aabc -> False"
    assert has_unique_characters(MockTextInput("")) == True, "Пустая строка -> True"
    assert has_unique_characters(MockTextInput("a")) == True, "Один символ -> True"
    print("✅ Все тесты пройдены\n")
    
    # Задача 3: Единичные биты
    print("Тесты для count_one_bits:")
    assert count_one_bits(MockPositiveIntegerInput(5)) == 2, "5 (101) -> 2"
    assert count_one_bits(MockPositiveIntegerInput(7)) == 3, "7 (111) -> 3"
    assert count_one_bits(MockPositiveIntegerInput(0)) == 0, "0 -> 0"
    assert count_one_bits(MockPositiveIntegerInput(255)) == 8, "255 (11111111) -> 8"
    print("✅ Все тесты пройдены\n")
    
    # Задача 4: Мультипликативная устойчивость
    print("Тесты для multiplicative_persistence:")
    assert multiplicative_persistence(MockPositiveIntegerInput(39)) == 3, "39 -> 3"
    assert multiplicative_persistence(MockPositiveIntegerInput(4)) == 0, "4 -> 0"
    assert multiplicative_persistence(MockPositiveIntegerInput(999)) == 4, "999 -> 4"
    assert multiplicative_persistence(MockPositiveIntegerInput(1)) == 0, "1 -> 0"
    print("✅ Все тесты пройдены\n")
    
    # Задача 5: MSE
    print("Тесты для mse:")
    assert mse(MockVectorPairInput([1, 2], [1, 2])) == 0.0, "Идентичные -> 0"
    assert mse(MockVectorPairInput([1, 2], [2, 3])) == 1.0, "(1,2) vs (2,3) -> 1"
    assert mse(MockVectorPairInput([0, 0], [1, 1])) == 1.0, "(0,0) vs (1,1) -> 1"
    print("✅ Все тесты пройдены\n")
    
    # Задача 6: Разложение на простые множители
    print("Тесты для prime_factorization:")
    assert prime_factorization(MockPositiveIntegerInput(86240)) == "(2**5)(5)(7**2)(11)", "86240"
    assert prime_factorization(MockPositiveIntegerInput(10)) == "(2)(5)", "10"
    assert prime_factorization(MockPositiveIntegerInput(7)) == "(7)", "7 (простое)"
    assert prime_factorization(MockPositiveIntegerInput(1)) == "", "1 -> пустая строка"
    print("✅ Все тесты пройдены\n")
    
    # Задача 7: Пирамида из кубиков
    print("Тесты для pyramid:")
    assert pyramid(MockPositiveIntegerInput(1)) == 1, "1 -> 1"
    assert pyramid(MockPositiveIntegerInput(5)) == 2, "5 (1+4) -> 2"
    assert pyramid(MockPositiveIntegerInput(14)) == 3, "14 (1+4+9) -> 3"
    assert pyramid(MockPositiveIntegerInput(10)) == "It is impossible", "10 -> невозможно"
    print("✅ Все тесты пройдены\n")
    
    # Задача 8: Сбалансированное число
    print("Тесты для is_balanced_number:")
    assert is_balanced_number(MockPositiveIntegerInput(1234006)) == True, "1234006 -> True"
    assert is_balanced_number(MockPositiveIntegerInput(123456)) == False, "123456 -> False"
    assert is_balanced_number(MockPositiveIntegerInput(121)) == True, "121 -> True"
    assert is_balanced_number(MockPositiveIntegerInput(11)) == True, "11 -> True"
    print("✅ Все тесты пройдены\n")
    
    print("🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
