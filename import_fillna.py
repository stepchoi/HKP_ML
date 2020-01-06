class c:
    def __init__(self, list):
        self.lst = list + ['init']

    def a(self):
        self.lst.extend('a')
        return self.lst

    def b(self):
        self.lst.extend('b')
        return self.lst


if __name__ == "__main__":
    lst_main = []
    class_main = c(lst_main).b()
    print(class_main)

    pass