class Person():
    def __init__(self, name, age):
        self.name = name;
        self.age = age;

    def greeting(self):
        print "Hello", self.name
        print "Your age is", self.age


P1 = Person("Amir", 38)
P1.greeting()

print P1.name
