from random import choice


class Person:

    def __init__(self, name):
        self.greeting = '<div>Hello {name}<div>'
        self.name = name

    def __str__(self):
        return self.make_greeting()

    def make_greeting(self):
        return self.greeting.format(name=self.name)


def main():
    people = [
        Person('Jane'),
        Person('Amir'),
        Person('Haiam')
    ]
    person = choice(people)
    print(person)


if __name__ == '__main__':
    main()
