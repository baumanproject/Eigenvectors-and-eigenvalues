import numpy as np
# размеры области
XSize = 200
YSize = 100
# Получаем количество элементов
X = 3#int(input('X: '))
Y = 2#int(input('Y: '))
# Проводим триангуляцию
vertices = np.zeros(((X + 1) * (Y + 1), 2))
elements = np.zeros((X * Y * 2, 3), dtype = np.int)
u_border = [] # все узлы, принадлежащие границе с ГУ 1-го рода
u_border_x = [] # часть границы, параллельная OX
u_border_y = [] # часть границы, параллельная OY
t_border = [] # все узлы, принадлежащие ГУ 3-го рода
for i in range(X + 1):
    for j in range(Y + 1):
        k = i * (Y + 1) + j # номер вершины
        vertices[k, 0] = i * XSize / X;
        vertices[k, 1] = j * YSize / Y;
        if i == 0 or j == 0 or j == Y:
            u_border.append(k)
            if j == 0 or j == Y:
                u_border_x.append(k)
                if i == 0:
                    u_border_y.append(k)
        if i == X:
            t_border.append(k)
print( u_border_x)
print(u_border_y)
#print()
for i in range(X):
    for j in range(Y):
        k = 2 * (i * Y + j)

        elements[k, 0] = i * (Y + 1) + j
        elements[k, 1] = i * (Y + 1) + j + 1 # вдоль oy
        elements[k, 2] = (i + 1) * (Y + 1) + j # вдоль ox

        elements[k + 1, 0] = (i + 1) * (Y + 1) + j + 1
        elements[k + 1, 1] = i * (Y + 1) + j + 1 # вдоль oy
        elements[k + 1, 2] = (i + 1) * (Y + 1) + j # вдоль ox

# составляем С
E = 212e9 # модуль Юнга
v = 0.29 # коэф. Пуассона
mu =0#1000
t = 1e5
# рассчитываем параметры Ламе
l1 = E*v/((1 + v)*(1 - 2*v))
l2 = E/(2*(1+v))
C1111 = l1 + 2 * l2
C1122 = l1
C1112 = 0
C2222 = l1 + 2 * l2
C2212 = 0
C1212 = l2
C = np.array([[C1111, C1122, C1112],
 [C1122, C2222, C2212],
 [C1112, C2212, C1212]])

def vec_len(a):
    return np.sqrt(np.sum(a ** 2))
# Составляем локальную СЛАУ
def make_local_slae(k, C, mu, u_border, t, t_border, elements, vertices):
    k0, k1, k2 = elements[k] # получаем номера вершин

# считаем площадь элемента k
    vec_a = vertices[k1] - vertices[k0]
    vec_b = vertices[k2] - vertices[k0]
    vec_c = vertices[k2] - vertices[k1]

    S_k = np.abs(0.5 * np.cross(vec_a, vec_b))

# ----------------
# ищем G_k
# ----------------
# cчитаем матрицу B
    phi1_1 = vec_a[1]/(vec_a[0]*vec_b[1] - vec_a[1]*vec_b[0]) - vec_b[1]/(vec_a[0]*vec_b[1] - vec_a[1]*vec_b[0])
    phi1_2 = -vec_a[0]/(vec_a[0]*vec_b[1] - vec_a[1]*vec_b[0]) + vec_b[0]/(vec_a[0]*vec_b[1] - vec_a[1]*vec_b[0])
    phi2_1 = vec_b[1]/(vec_a[0]*vec_b[1] - vec_a[1]*vec_b[0])
    phi2_2 = -vec_b[0]/(vec_a[0]*vec_b[1] - vec_a[1]*vec_b[0])
    phi3_1 = -vec_a[1]/(vec_a[0]*vec_b[1] - vec_a[1]*vec_b[0])
    phi3_2 = vec_a[0]/(vec_a[0]*vec_b[1] - vec_a[1]*vec_b[0])

    B = np.array([[phi1_1, 0, phi2_1, 0, phi3_1, 0],
                    [0, phi1_2, 0, phi2_2, 0, phi3_2],
                    [phi1_2, phi1_1, phi2_2, phi2_1, phi3_2, phi3_1]])

    #print(B)

    G_k_1 = np.dot(np.dot(np.transpose(B), C), B) * S_k;
    #print(G_k_1)
    G_k_2 = np.zeros((3, 3))

    if k0 in u_border and k1 in u_border:
        L = vec_len(vec_a)
        G_k_2 += mu * np.array([[L/4, L/12, L/12], [L/12, L/12, 0], [L/12, 0, L/12]])
        G_k_2 += mu * np.array([[L/12, L/12, 0], [L/12, L/4, L/12], [0, L/12, L/12]])
    if k0 in u_border and k2 in u_border:
        L = vec_len(vec_b)
        G_k_2 += mu * np.array([[L/4, L/12, L/12], [L/12, L/12, 0], [L/12, 0, L/12]])
        G_k_2 += mu * np.array([[L/12, 0, L/12], [0, L/12, L/12], [L/12, L/12, L/4]])
    if k1 in u_border and k2 in u_border:
        L = vec_len(vec_c)
        G_k_2 += mu * np.array([[L/12, L/12, 0], [L/12, L/4, L/12], [0, L/12, L/12]])
        G_k_2 += mu * np.array([[L/12, 0, L/12], [0, L/12, L/12], [L/12, L/12, L/4]])



    G_k_3 = np.kron(G_k_2, np.eye(2))

    G_k = G_k_1 + G_k_3

    #print(G_k)
# -------------------
# ищем f_k
# -------------------
    L = 0
    if k0 in t_border and k1 in t_border:
            L += vec_len(vec_a)
    if k0 in t_border and k2 in t_border:
            L += vec_len(vec_b)
    if k1 in t_border and k2 in t_border:
            L += vec_len(vec_c)

    t_eh = np.zeros((6))
    if k0 in t_border:
        t_eh[0] = -t
    if k1 in t_border:
        t_eh[2] = -t
    if k2 in t_border:
        t_eh[4] = -t

    if not k0 in t_border:
        f_k_0 = [[0, 0, 0], [0, L/3, L/6], [0, L/6, L/3]]
    elif not k1 in t_border:
        f_k_0 = [[L/3, 0, L/6], [0, 0, 0], [L/6, 0, L/3]]
    else:
        f_k_0 = [[L/3, L/6, 0], [L/6, L/3, 0], [0, 0, 0]]

    f_k_1 = np.kron(f_k_0, np.eye(2))
    f_k= np.dot(f_k_1, t_eh)
    #print(f_k)
    return G_k, f_k
#сборка глобальной СЛАУ
u = np.zeros((2 * vertices.shape[0]))
f = np.zeros((2 * vertices.shape[0]))
G = np.zeros((2 * vertices.shape[0], 2 * vertices.shape[0]))
def gl_indexes(k, elements):
    k0, k1, k2 = elements[k]
    return [2 * k0, 2 * k0 + 1, 2 * k1, 2 * k1 + 1, 2 * k2, 2 * k2 + 1]

# учет правой части локальных СЛАУ
for k in range(2 * X * Y):
    G_k, f_k = make_local_slae(k, C, mu, u_border, t, t_border, elements, vertices)
    g = gl_indexes(k, elements)
    for i in range(6):
        f[g[i]] += f_k[i]

# учет граничных условий 1-го рода
for i in u_border_x:
     k = 2 * i + 1
     G[k, k] = 1
     f[k] = 0

for i in u_border_y:
     k = 2 * i
     G[k, k] = 1
     f[k] = 0
print(G)
def is_border(k, i, u_border_x, u_border_y):
    if i % 2 == 0 and elements[k, i // 2] in u_border_y:
        return True
    elif i % 2 == 1 and elements[k, i // 2] in u_border_x:
        return True
    return False
# учет матриц локальных СЛАУ
for k in range(2 * X * Y):
    G_k, f_k = make_local_slae(k, C, mu, u_border, t, t_border, elements, vertices)
    g = gl_indexes(k, elements)
    for i in range(6):
        if is_border(k, i, u_border_x, u_border_y):
            continue
        for j in range(6):
            if not is_border(k, j, u_border_x, u_border_y):
                G[g[i], g[j]] += G_k[i, j]
            else:
                f[g[i]] -= G_k[i, j] * f[g[j]]

# решение глобальной СЛАУ
u = np.linalg.solve(G, f)

# вывод результата
def save_result(u, vertices, elements):
    with open('result.mv2', 'w') as file:
        file.write('{0} 3 2 UX UY\n'.format(vertices.shape[0]))
        for i in range(vertices.shape[0]):
            u_x = u[i * 2]
            u_y = u[i * 2 + 1]
            x = vertices[i, 0]
            y = vertices[i, 1]
            file.write('{0} {1} {2} 0 {3:.16f} {4:.16f}\n'.format(i + 1, x, y, u_x, u_y))
        file.write('{0} 3 3 BC_id mat_id mat_id_Out\n'.format(elements.shape[0]))
        for i in range(elements.shape[0]):
            k0, k1, k2 = elements[i]
            file.write('{0} {1} {2} {3} 0 1 0\n'.format(i + 1, k0 + 1, k1 + 1, k2 + 1))

save_result(u, vertices, elements)
