import numpy as np

"""Inspired by https://github.com/bluekeybo/AES"""

class AES:  # for 16 bits
    def __init__(self):
        self.rounds= ... # how is this number determined?
        self.S_box = ...    # Add lookup table matrix here that maps things nonlinearly to other things.

    
    def SubBytes(self, state):
        return self.S_box[state]

    def ShiftRows(self, state):
        ...

    def MixColumns(self, state):
        ...

    def AddRoundKey(self, state, key):
        return np.bitwise_xor(state, key)

    def encrypt(self, plaintext):
        # create state here first
        state = ...

        for i in range(1, self.rounds):
            state = self.SubBytes(state=state)
            state = self.ShiftRows(state=state)
            state = self.MixColumns(state=state)
            state = self.AddRoundKey(state=state, key=self.keys[i])

        # MixColumns is not used on the final round
        state = self.SubBytes(state=state)
        state = self.ShiftRows(state=state)
        state = self.AddRoundKey(state=state, key=self.keys[self.rounds])

        ciphertext = ...

        return ciphertext

    def decrypt(self, ciphertext):
        ...

def main():
    plaintext = [[0, 1, 1, 0],
                 [1, 1, 0, 1], 
                 [0, 0, 0, 1], 
                 [1, 0, 1, 0]]
    
    aes = AES()

    ciphertext = aes.encrypt(plaintext)

if __name__ == "__main__":
    main()