# paint app
try:
    from point import Point
    from rect import Rect
except ImportError as e:
    print(e)


def test_bound_instance_method():
    p = Point()
    p.set_x(10)
    p.set_y(20)
    # print(p.get_x(), p.get_y(), sep=',')
    p.show()


def test_unbound_class_method():
    p = Point()
    Point.set_x(p, 10)
    Point.set_y(p, 20)
    print(Point.get_x(p), Point.get_y(p), sep=',')


def test_member():
    p = Point()
    p.set_x(10)
    p.set_y(10)
    print('x={0}, y={1}, count_of_instance={2}'.format(p.x, p.y, p.count_of_instance))


def test_constructor():
    p1 = Point(10, 20)
    print('x={0}, y={1}, count_of_instance={2}'.format(p1.x, p1.y, Point.get_count_of_instance()))

    p2 = Point(100, 200)
    print('x={0}, y={1}, count_of_instance={2}'.format(p2.x, p2.y, Point.get_count_of_instance()))

    del p1
    print('count_of_instance={0}'.format(Point.get_count_of_instance()))

    del p2
    print('count_of_instance={0}'.format(Point.get_count_of_instance()))


def test_to_string():
    p = Point()
    print(p)
    print(repr(p))

    p2 = eval(repr(p))
    print(p2)


def test_class_rect():
    r1 = Rect(10, 10, 50, 50)
    r2 = eval(repr(r1))

    print(r1, str(r1.area()), sep=':')
    print(r2, str(r2.area()), sep=':')

    r1.draw()
    r2.draw()


def test_oerator_overloading():
    p1 = Point(100, 200)
    p2 = Point(50, 50)
    p3 = p1 + p2
    p4 = p1 - p2

    print(p3)
    print(p4)

    p3 += Point(-10, -10)
    p4 -= Point(-10, -10)

    print(p3)
    print(p4)

    print(Rect(10, 20) == Rect(20, 10))
    print(Rect(10, 10) > Rect(5, 10))
    print(Rect(10, 20) < Rect(20, 10))

def main():
    # test_unbound_class_method()
    # test_bound_instance_method()
    # test_member()
    # test_constructor()
    # test_to_string()
    # test_class_rect()
    test_oerator_overloading()


if __name__ == '__main__':
    main()
