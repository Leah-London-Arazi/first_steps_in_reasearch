import numpy as np
import pandas as pd

# bits methods
def remove_digit(x, digit):
    l = x.split(",")
    if str(digit) in l:
        l.remove(str(digit))
    return ",".join(l)

def compare_bits(bit, other_bit):
    '''
    returns 0 if equal.
    returns 1 if bit > other_bit.
    returns -1 if bit < other_bit.
    returns None if the bits are stable and different.
    '''
    if bit == other_bit:
        return 0
    if bit == 'u':
        return -1
    if other_bit == 'u':
        return 1
    return None

def diff_digits(term1, term2):
    return [i for i in range(len(term1)) if compare_bits(term1[i], term2[i]) == None]

def is_contained(term1, term2):
    '''
    returns 1 if term1 contained in term2 (term2 -> term1).
    returns -1 if term2 contained in term1 (term1 -> term2).
    returns 0 if equal.
    else None.
    '''
    res = 0
    for i in range(len(term1)):
        comp = compare_bits(term1[i], term2[i])
        if comp != 0:
            if comp != res:
                if res != 0:
                    return None
                else:
                    res = comp
    return res

def is_complement(term1, term2):
    '''
    returns the different stable bit if term1 is contained in term2 at all bits except one.
    otherwise returns -1.
    '''
    diff = diff_digits(term1, term2)
    if len(diff) == 1:
        for i in range(len(term1)):
            if i != diff[0] and compare_bits(term1[i], term2[i]) != 0:
                return -1
        return diff[0]
    return -1

def remove_literal(term, digit):
    return term[:digit] + 'u' + term[digit+1:]

def substitute(term, stable_bits):
    bits = list(term)
    j = 0
    for i, bit in enumerate(term):
        if bit == 'u':
            bits[i] = stable_bits[j]
            j += 1
    return "".join(bits)

def all_resolutions(term):
    bits = list(term)
    u_count = bits.count('u')
    stable_format = "{" + "0:0{}b".format(u_count) + "}"
    return [substitute(term, stable_format.format(i)) for i in range(2 ** u_count)]

class BooleanFunction(object):
    def __init__(self, n, one_inputs):
        """
        n - the number of inputs bits in the function
        one_inputs - a list of strings. the function calculates 1 iff the input is in that list.
        """
        self.n = n
        self.formatting = "{" + "0:0{}b".format(n) + "}"
        self.one_inputs = one_inputs
        self.zero_inputs = [self.formatting.format(i) for i in range(2**n) if self.formatting.format(i) not in set(one_inputs)]
        self.full_matrix = self.__create_full_matrix()
        self.partial_matrix = self.__create_partial_matrix()
        self.id = int("".join(['1' if self.formatting.format(i) in set(one_inputs) else '0' for i in range(2**n)][::-1]), 2)

    def is_monotone(self):
        '''
        for all x,y if x<=y then f(x)<=f(y).
        or, not(exists x<=y such that f(x)>f(y)).
        '''
        zeros = set(self.zero_inputs)
        for one_input in self.one_inputs:
            for i, bit in enumerate(one_input):
                if bit == '0':
                    if (one_input[:i] + '1' + one_input[i+1:]) in zeros:
                        return False
        return True

    def calc(self, input):
        return 1 if input in self.one_inputs else 0

    def __create_matrix(self, rows, columns):
        table = []
        for term1 in rows:
            row = []
            for term2 in columns:
                row.append(str(diff_digits(term1, term2))[1:-1])
            table.append(row)

        x = np.array(table)
        if len(rows) == 0 or len(columns) == 0:
            return pd.DataFrame(x)
        return pd.DataFrame(x, index=rows, columns=columns)

    def __create_full_matrix(self):
        return self.__create_matrix(self.one_inputs, self.zero_inputs)

    def merge_terms(self, terms):
        prime_terms = list(terms)
        i = 0
        while i < len(prime_terms):
            drop_flag = False
            restart_flag = False
            for j in range(len(prime_terms)):
                result = self.merge(prime_terms[i], prime_terms[j])
                if result:
                    new_term, drop_term, diff_digit = result
                    if new_term: # continue if a new term was created
                        restart_flag = True
                        if new_term not in prime_terms:
                            prime_terms.append(new_term)
                    if drop_term == prime_terms[i]:
                        restart_flag = True
                        drop_flag = True

            if drop_flag:
                prime_terms.remove(prime_terms[i])
            if restart_flag:
                i = 0
            else:
                i += 1

        return prime_terms

    def merge(self, term1, term2):
        '''
        new_term, drop, diff
        '''
        digit = is_complement(term1, term2)
        if digit != -1:  # complement, we can drop term1 and add u
            return remove_literal(term1, digit), term1, digit
        res = is_contained(term1, term2)
        if res == -1:  # we can drop term2
            return None, term2, None
        if res == 1:  # we can drop term1
            return None, term1, None
        return None # cannot merge

    def __create_partial_matrix(self):
        # merge rows
        one_terms = self.merge_terms(self.one_inputs)
        # merge cols
        zero_terms = self.merge_terms(self.zero_inputs)
        return self.__create_matrix(one_terms, zero_terms)

    def get_derivative(self, x, y):
        term = "".join(['u' if y[i] == '1' else x[i] for i in range(len(x))])
        resolutions = all_resolutions(term)
        f_on_x = self.calc(x)
        for res in resolutions:
            if self.calc(res) != f_on_x:
                return 1
        return 0

    def get_derivative_as_function(self, x):
        der = []
        for i in range(2 ** self.n):
            y = self.formatting.format(i)
            if self.get_derivative(x, y) == 1:
                der.append(y)
        return BooleanFunction(self.n, der)

    def get_all_derivatives(self):
        all_der = []
        for i in range(2 ** self.n):
            x = self.formatting.format(i)
            all_der.append(self.get_derivative_as_function(x))
        return all_der

def str_communication_matrices(f, header, as_html=True):
    string = "################ {} ################".format(header) + "\n" + \
             (f.full_matrix.to_html() if as_html else f.full_matrix) + "\n\n" + \
             (f.partial_matrix.to_html() if as_html else f.partial_matrix) + "\n"
    return string

def all_possible_matrices(n):
    functions = []
    formatting = "{" + "0:0{}b".format(n) + "}"
    func_formatting = "{" + "0:0{}b".format(2 ** n) + "}"
    inputs = [formatting.format(i) for i in range(2 ** n)]
    for i in range(2 ** (2 ** n)):
        func = func_formatting.format(i)[::-1]
        f = BooleanFunction(n, [inputs[k] for k in range(2 ** n) if func[k] == '1'])
        functions.append(f)
    return functions

def print_all_matrices(n, monotone=False, to_file=False, file_path="", to_print_functions=[]):
    func_formatting = "{" + "0:0{}b".format(2 ** n) + "}"
    functions = all_possible_matrices(n) if len(to_print_functions) == 0 else to_print_functions
    string = "\n".join([str_communication_matrices(f, str(f.id) + " | " + func_formatting.format(f.id), as_html=to_file) for f in functions if not(f.is_monotone()) or monotone])
    if to_file:
        with open(file_path, "w") as file:
            file.write(string)
    else:
        print(string)

def main():
    '''
    set output file dir
    '''
    output_files_dir = "."
    '''
    print all matrices for n=3
    '''
    # print_all_matrices(3, monotone=True, to_file=True, file_path=output_files_dir + "\\all_functions_3_inputs_with_monotone.html")
    # print_all_matrices(3, monotone=False, to_file=True, file_path=output_files_dir + "\\all_functions_3_inputs_no_monotone.html")
    '''
    print only monotone matrices for n=3
    '''
    # functions = all_possible_matrices(3)
    # mono_functions = []
    # for i, f in enumerate(functions):
    #      if f.is_monotone():
    #          mono_functions.append(f)
    # print_all_matrices(3, monotone=True, to_file=True, file_path=output_files_dir + "\\monotone_functions_3_inputs.html", to_print_functions=mono_functions)
    '''
    print derivative matrices for a given function
    '''
    # f = BooleanFunction(3, ['100', '110'])
    # all_der = f.get_all_derivatives()
    # print_all_matrices(3, monotone=True, to_file=True,
    #                    file_path=output_files_dir + "\\all_derivatives_of_f_3.html",
    #                    to_print_functions=[f] + all_der)

    # f = BooleanFunction(5, ['10101', '01010', '11111', '00001'])
    # all_der = f.get_all_derivatives()
    # print_all_matrices(5, monotone=True, to_file=True,
    #                    file_path=output_files_dir + "\\all_derivatives_of_f_5.html",
    #                    to_print_functions=all_der)

if __name__ == '__main__':
    main()