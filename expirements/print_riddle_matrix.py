import numpy as np
import pandas as pd

class ProductTerm(object):
    def __init__(self):
        pass
    def diff_digits(self):
        pass
    def is_complement(self, other):
        '''
        if complement will return the different digit, else 1
        '''
        pass
    def is_contained(self, other):
        pass
    def remove_literal(self, digit):
        pass
    def merge(self, other):
        '''
        new_input, drop_input, diff_digit
        '''
        pass
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

    def __add__(self, other):
        sum_diff = DiffDigits([])
        digits = self.digits | other.digits
        removed_digits = self.removed_digits | other.removed_digits
        sum_diff.removed_digits = removed_digits

        for d in digits:
            if d not in removed_digits:
                sum_diff.digits.add(d)

        return sum_diff

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
        self.one_inputs = one_inputs
        self.zero_inputs = [self.formatting.format(i) for i in range(2**n) if self.formatting.format(i) not in set(one_inputs)]
        self.full_matrix = self.__create_full_matrix()

    def calc(self, input):
        return 1 if input in self.one_inputs else 0

    def __create_full_matrix(self):
        table = []
        for one_input in self.one_inputs:
            row = []
            for zero_input in self.zero_inputs:
                row.append(self.__find_diff_digits(one_input, zero_input))
            table.append(row)

        x = np.array(table)
        return pd.DataFrame(x, index=self.one_inputs, columns=self.zero_inputs)

    def __find_diff_digits(self, input, other_input):
        return DiffDigits([i+1 for i in range(self.n) if input[i] != other_input[i] and input[i] != 'u' and other_input[i] != 'u'])

    def find_merge(self, df, axis=1):
        inputs = df.index if axis == 1 else df.columns
        for input1 in inputs:
            for input2 in inputs:
                res = self.merge(input1, input2)
                if res:
                    new_input, drop_input, diff_digit = res
                    series = df.loc[drop_input] if axis == 1 else df[drop_input]
                    new_series = series.apply(lambda x: x.remove(diff_digit))
                    return input1, input2, new_input, new_series, drop_input
        return None

    def compare_us(self, input1, input2):
        '''
        0 - equal
        1 - input1 is bigger
        -1 - input2 is bigger
        None - incompatible
        '''
        res = 0
        for i in range(self.n):
            if input1[i] == 'u' and input2[i] != 'u':
                if res == 0:
                    res = -1
                if res == 1:
                    return None
            if input1[i] != 'u' and input2[i] == 'u':
                if res == 0:
                    res = 1
                if res == -1:
                    return None
        return res

    def merge(self, input1, input2):
        '''
        new_input, drop, diff
        '''
        diff = self.__find_diff_digits(input1, input2)
        if len(diff) > 1:
            return None
        comp_u = self.compare_us(input1, input2)
        if comp_u == None:
            return None
        '''
        inputs can be complement/included
        '''
        if len(diff) == 0: # included
            if comp_u == 1: # input2 has more u's
                return input2, input1, None
            if comp_u == -1: # input1 has more u's
                return input1, input2, None
        else: # complement
            if comp_u == 1 or comp_u == 0:  # input2 has more u's
                return input1[:diff.get_digit()-1] + 'u' + input1[diff.get_digit():], input1, diff.get_digit()
            if comp_u == -1:  # input1 has more u's
                return input2[:diff.get_digit()-1] + 'u' + input2[diff.get_digit():], input2, diff.get_digit()

        return None

    def get_merged_matrix(self):
        df = self.full_matrix
        # merge rows
        res = self.find_merge(df, axis=1)
        while res:
            input1, input2, new_input, new_series, drop_input = res
            df = df.drop(index=list([drop_input]))
            df.loc[new_input] = new_series
            res = self.find_merge(df, axis=1)
        # transpose?!?
        # merge cols
        res = self.find_merge(df, axis=0)
        while res:
            input1, input2, new_input, new_series, drop_input = res
            df = df.drop(columns=list([drop_input]))
            df[new_input] = new_series
            res = self.find_merge(df, axis=0)
        return df

def main():
    f = BooleanFunction(3, ['001', '101', '110'])
    f.get_merged_matrix()

if __name__ == '__main__':
    main()