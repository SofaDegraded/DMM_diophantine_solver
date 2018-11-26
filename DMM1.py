import numpy as np
import sympy as sm
import os

'''
Считывание входных данных
Input:
flag - по умолчанию идет считывание из файла
иначе с консоли
Return: 
n - число уравнений
m - число неизвестных переменных
eq - коэффициенты уравнений
'''
def reading(flag='file'):
    if flag == 'file':
        path = os.path.dirname(os.path.realpath(__file__))
        filename = path + '/input.txt'
        with open(filename) as f:
            n, m = [int(x) for x in next(f).split()]
            eq = np.array([[int(x) for x in line.split()] for line in f])
            eq[:,-1] = - eq[:,-1]
            if any(x is None for x in list(map(lambda x: print('ValueError') if x != m + 1  else x, list(map(len, eq))))):
                raise ValueError('Введено неверное число коэффициентов!')
    else:
        n = int(input('Введите число уравнений: '))
        m = int(input('Введите число неизвестных переменных: '))
        eq = [list(map(int, input('Введите коэффициенты уравнения %s:' %(i + 1)).split())) for i in range(n)]
        if any(x is None for x in list(map(lambda x: print('ValueError') if x != m  else x, list(map(len, eq))))):
            raise ValueError('Введено неверное число коэффициентов!')
        c = [- int(input('Введите правую часть уравнения %s:' %(i + 1))) for i in range(n)]
        eq = np.column_stack([eq, c])
    return n, m, eq.tolist()

'''
Печать результата на экран или в файл
Input:
B - преобразованная матрица с решением уравнения
Return: 
NONE
'''
def print_and_write_result(B, flag='file'):
    if flag == 'file':
        file = open('output.txt', 'w')
        file.write(str(len(B[0]) - 1) + '\n')
        for r in B:
            file.write(' '.join(map(str,r)) + '\n')
        file.close()
    else:
        print('Свободные переменные:',str(len(B[0]) - 1))
        for r in B:
            print(' + t *'.join(map(str,r)))

'''
Построение матрицы В для случая решений
линейного уравнения и для решения СЛДУ
Input:
n - число уравнений
m - число неизвестных переменных
eq - коэффициенты уравнений
Return: 
матрица В
'''
def build_matrix_B(n, m, eq,n_new=None):
    #Единичная матрица
    I = np.eye(m)
    #Если решается система, то еще добавляем нулевой вектор
    if n > 1:
        zero_vec = np.zeros(m)
    #Строим матрицу В
    B = list(map(np.ndarray.tolist, np.concatenate([eq, np.column_stack((I,zero_vec))]) \
            if n > 1 or n_new!=None else np.row_stack([eq[0][:-1], I])))
    return B

'''
Удаление нулевых и линейно-зависимых строк
'''
def del_null_and_lin_dep_row(coef_m, n, m):
    # Удаляем нулевые строки
    coef_m = [i for i in coef_m if i != [0] * m]
    if np.linalg.matrix_rank(coef_m) != n:
        #Удаляем линейно зависимые строки
        _, inds = sm.Matrix(coef_m).T.rref()
        coef_m = np.array(coef_m)[[inds]].tolist()
        n = len(coef_m)
    return n, coef_m

'''
'''
def check_elem(B):
    was_found = False
    for i in B[0]:
        if i != 0:
            if was_found:
                return False
            was_found = True
    return True
'''
Решение диофантовых уравнений
(в случае 1 уравнения и системы)
Input:
n - число уравнений
m - число неизвестных переменных
eq - коэффициенты уравнений
Return: 
матрица В приведенная к эквивалентному виду
(треугольная)
'''
def diophantine_equation_solver(n, m, eq):
    #Составляем матрицу B
    #Решаем линейное диофантовое уравнение
    if n == 1:
        B = build_matrix_B(n, m, eq)
        flag = True
        while flag:
            #Шаг 1.
            #Выбираем в первой строке матрицы В наименьший
            #по абсолютной величине ненулевой элемент a[i]
            min_el = min(map(abs, B[0]))
            min_ind = list(map(abs, B[0])).index(min_el)
            for j in range(m):
            #Шаг 2.
            #Выбираем номер j!=i такой, что a[j]!=0
                if j != min_ind and B[0][j]!= 0:
                    #Шаг 3.
                    #Делим с остатком на a[j], т.е. находим q и r, что 
                    #a[j]=qa[i]+r
                    q = B[0][j] // B[0][min_ind]
                    for k in range(m+n):
                        #Шаг 4.
                        #Вычитаем из j-го столбца матрицы В i-й столбец
                        #умноженный на q
                        B[k][j] -= q * B[k][min_ind]
            #Шаг 5.
            #Если в первой строке более одного ненулевого числа, то выход,
            #иначе переходим на шаг 1
            if B[0].count(0) >= 1 or len(B[0]) == 1:
                flag = False
            else:
                flag = True
        #Свободный член уравнения
        c = - eq[0][-1]
        #Вычисляем индекс, где находится d=НОД()
        max_el = max(map(abs, B[0]))
        max_ind = list(map(abs, B[0])).index(max_el)
        #d=НОД()
        d = B[0][max_ind]
        #Если делится без остатка, то
        #вычисляем коэффициент  
        #и выводим результат
        if c % d == 0:
            coef = c / d
            for i in range(1, n + m):
                B[i][max_ind] = coef * B[i][max_ind]
            tmp_b = np.array(B)
            B = np.c_[tmp_b[:,max_ind], np.delete(tmp_b, max_ind, 1)]
            B = np.delete(B, 0, 0).astype(int).tolist()
        #Иначе решений в целых числах нет
        else:
            raise ValueError('Решений в целых числах нет!')    

    #Решаем систему линейных диофантовых уравнений
    else:
        #Удаляем нулевые и линейно-зависимые строки
        n_new, eq = del_null_and_lin_dep_row(eq, n, m)
        #Строим матрицу В
        B = build_matrix_B(n, m, eq, n_new=n_new)
        m_new = len(eq)
        #Приводим матрицу к виду трапеции
        for i in range(m_new):
            b = [(B[k][i:-1]) for k in range(i, len(B))]
            b = np.array(b).astype(int).tolist()
            while  check_elem(b) == False:
                min_el = list(map(lambda x: abs(x) if x != 0 else 1000, b[0]))
                i_min = min_el.index(min(min_el))
                max_el = list(map(lambda x: abs(x) if x != 0 else -1000, b[0]))
                i_max = len(max_el) - max_el[::-1].index(max(max_el))-1
                coef = b[0][i_max] // b[0][i_min]
                #Вычитаем из imax-го столбца матрицы В imin-й столбец
                #умноженный на coef
                for k in range(len(b)):
                    b[k][i_max] -= coef * b[k][i_min]
            if not b[0]:
                raise ValueError('Нет решения!')
            max_el = list(map(lambda x: abs(x) if x != 0 else -1000, b[0]))
            col = len(max_el) - max_el[::-1].index(max(max_el)) - 1
            b = [[row[col]] + row[:col] + row[col+1:] for row in b]
            l = len(B)
            B = [B[k] if k < i else B[k][0:i] + b[k - i] + [int(B[k][-1])] for k in range(l)]
            #Проверка на отсутствие нулей на диагонали
            if B[i][i] == 0:
                raise ValueError('Нет решения!')
            else:
                coef = B[i][-1] // B[i][i]
            #Вычитаем из последнего столбца матрицы В i-й столбец
            #умноженный на coef
            for k in range(len(B)):
                B[k][-1] -= (coef * B[k][i])
        #Проверка, что в последний стлбец обнулился
        if [np.array(B)[i,-1] for i in range(m_new)] != m_new * [0]:
            raise ValueError('Нет решения!')
        size = len(eq[0]) - 1
        #Приводим к виду для печати результата
        tmp_b = np.array([B[m_new + i][m_new:] for i in range(size)])
        B = np.c_[tmp_b[:,-1], np.delete(tmp_b, -1, 1)]
    return B

if __name__ == '__main__':
    n, m, eq = reading()
    res = diophantine_equation_solver(n, m, eq)
    print_and_write_result(res, flag='file')