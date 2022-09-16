specializing = False

class Specializer:
    @staticmethod
    def enter():
        print("Enter")
        global specializing
        specializing = True

    @staticmethod
    def exit():
        print("Exit")
        global specializing
        specializing = False

def specialize():
    return Specializer()
