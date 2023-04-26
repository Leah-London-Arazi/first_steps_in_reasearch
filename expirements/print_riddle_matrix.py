import numpy as np
import pandas as pd

# bits methods
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
    return DiffDigits([i for i in range(len(term1)) if compare_bits(term1[i], term2[i]) == None])
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

def is_speciel_complement(term1, term2):
    '''
    returns the different stable bit if term1 is contained in term2 at all bits except one.
    otherwise returns -1.
    '''
    diff = diff_digits(term1, term2)
    if len(diff) == 1:
        for i in range(len(term1)):
            if compare_bits(term1[i], term2[i]) == -1:
                return -1
        return diff.get_digit()
    return -1

def remove_literal(term, digit):
    return term[:digit] + 'u' + term[digit+1:]

class DiffDigits(object):
    def __init__(self, digits):
        self.digits = set(digits)
        self.removed_digits = set()

    def remove(self, digit):
        if digit in self.digits and digit != None:
            self.digits.remove(digit)
            self.removed_digits.add(digit)
        return self

    def get_digit(self):
        return next(iter(self.digits))

    def __len__(self):
        return len(self.digits)

    def __str__(self):
        return str(self.digits)[1:-1]

class BooleanFunction(object):
    def __init__(self, n, one_inputs):
        """
        n - the number of inputs bits in the function
        one_inputs - a list of strings. the function calculates 1 iff the input is in that list.
        """
        self.n = n
        self.formatting = "{" + "0:0{}b".format(n) + "}"
        self.one_inputs = set(one_inputs)
        self.zero_inputs = [self.formatting.format(i) for i in range(2**n) if self.formatting.format(i) not in set(one_inputs)]
        self.full_matrix = self.__create_full_matrix()
        self.partial_matrix = self.__create_partial_matrix()

    def calc(self, input):
        return 1 if input in self.one_inputs else 0

    def __create_full_matrix(self):
        table = []
        for one_input in self.one_inputs:
            row = []
            for zero_input in self.zero_inputs:
                row.append(diff_digits(one_input, zero_input))
            table.append(row)

        x = np.array(table)
        return pd.DataFrame(x, index=self.one_inputs, columns=self.zero_inputs)

    def find_merge(self, df, axis=1):
        terms = df.index if axis == 1 else df.columns
        for term1 in terms:
            for term2 in terms:
                res = self.merge(term1, term2)
                if res:
                    new_term, drop_term, diff_digit = res
                    series = df.loc[drop_term] if axis == 1 else df[drop_term]
                    new_series = series.apply(lambda x: x.remove(diff_digit))
                    return term1, term2, new_term, new_series, drop_term
        return None

    def merge(self, term1, term2):
        '''
        new_term, drop, diff
        '''
        digit = is_speciel_complement(term1, term2)
        if digit != -1:  # spciel complement, we can drop term1 and add u
            return remove_literal(term1, digit), term1, digit
        digit = is_speciel_complement(term2, term1)
        if digit != -1:  # spciel complement, we can drop term2 and add u
            return remove_literal(term2, digit), term2, digit
        res = is_contained(term1, term2)
        if res == -1:  # we can drop term2
            return term1, term2, None
        if res == 1:  # we can drop term1
            return term2, term1, None
        return None # cannot merge

    def __create_partial_matrix(self):
        df = self.full_matrix
        # merge rows
        res = self.find_merge(df, axis=1)
        while res:
            term1, term2, new_term, new_series, drop_term = res
            df = df.drop(index=list([drop_term]))
            df.loc[new_term] = new_series
            res = self.find_merge(df, axis=1)
        # merge cols
        res = self.find_merge(df, axis=0)
        while res:
            term1, term2, new_term, new_series, drop_term = res
            df = df.drop(columns=list([drop_term]))
            df[new_term] = new_series
            res = self.find_merge(df, axis=0)
        return df

def main():
    f = BooleanFunction(3, ['111', '101', '010', '011']) # mux
    print(f.full_matrix)
    print(f.partial_matrix)
    # not working on mux!

if __name__ == '__main__':
    main()