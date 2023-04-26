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
                row.append(str(diff_digits(one_input, zero_input))[1:-1])
            table.append(row)

        x = np.array(table)
        return pd.DataFrame(x, index=self.one_inputs, columns=self.zero_inputs)

    def merge_rows_or_columns(self, df, axis=1):
        continue_flag = True
        while continue_flag:
            continue_flag = False
            terms = list(df.index) if axis == 1 else list(df.columns)
            for term1 in terms:
                drop_flag = False
                restart_flag = False
                for term2 in terms:
                    result = self.merge(term1, term2)
                    if result:
                        new_term, drop_term, diff_digit = result
                        if new_term: # continue if a new term was created
                            restart_flag = True
                            if axis == 1:
                                series = df.loc[drop_term].copy(deep=True)
                                df.loc[new_term] = series.apply(lambda x: remove_digit(x, diff_digit))
                            else:
                                series = df[drop_term].copy(deep=True)
                                df[new_term] = series.apply(lambda x: remove_digit(x, diff_digit))
                        if drop_term == term1:
                            drop_flag = True

                if drop_flag:
                    df = df.drop(index=list([term1])) if axis == 1 else df.drop(columns=list([term1]))
                if restart_flag:
                    continue_flag = True
                    break
        return df

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
        df = self.__create_full_matrix()
        # merge rows
        df = self.merge_rows_or_columns(df, axis=1)
        # merge cols
        df = self.merge_rows_or_columns(df, axis=0)
        return df

def main():
    f = BooleanFunction(3, ['111', '101', '010', '011']) # mux
    print(f.full_matrix)
    print(f.partial_matrix)
    # not working on mux!

if __name__ == '__main__':
    main()