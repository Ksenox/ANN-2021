def task10(number):

    string = str(number)

    lenStr = 0

    if len(string) < 3: return True

    if len(string) % 2 == 0:
        lenStr = len(string) / 2 - 1
    else:
        lenStr = (len(string) - 1) / 2

    leftValue = 0
    rightValue = 0

    for i in range(0, int(lenStr), 1):
        leftValue += int(string[i])
        rightValue += int(string[len(string) - i - 1])


    if leftValue == rightValue:
        return True

    return False


try:
    a = int(input("Input: "))

    if a < 0:
        print("Должно быть положительно число")
    else:
        print(task10(a))

except ValueError:
     print("Это не целое число! Выходим.")