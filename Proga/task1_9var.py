n = int(input("n="))
lst = [1]
for i in range(2, n+1):
    for j in range(2, i):
        if i % j == 0:
            # если делитель найден, число не простое.
            break
    else:
        lst.append(i)
print(lst)