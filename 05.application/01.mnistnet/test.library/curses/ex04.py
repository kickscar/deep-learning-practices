import curses
import time
import numpy as np


def curses_init():
    stdscr = curses.initscr()

    stdscr.keypad(True)
    curses.noecho()
    curses.cbreak()
    curses.curs_set(False)

    curses.start_color()
    curses.use_default_colors()

    curses.init_pair(1, curses.COLOR_GREEN, -1)
    curses.init_pair(2, curses.COLOR_YELLOW, -1)
    curses.init_pair(3, curses.COLOR_RED, -1)

    return stdscr,


def curses_cleanup(stdscr):
    curses.nocbreak()
    stdscr.keypad(False)
    curses.echo()
    curses.endwin()


def curses_prediction_result(stdscr, val, y):
    stdscr.addstr(0, 0, f'Model Prediction:', curses.color_pair(1))
    stdscr.addstr(0, 18, f'{val}', curses.color_pair(3) | curses.A_BOLD | curses.A_UNDERLINE)
    stdscr.addstr(0, 25, 'Class Probabilities', curses.color_pair(1))

    for idx, probablity in enumerate(np.round(y * 100., 2)):
        color_pair = curses.color_pair(3) if idx == val else curses.color_pair(2)
        filled_length = int(30 * probablity // 100)
        bar = '█' * filled_length + '-' * (30 - filled_length)

        stdscr.addstr(1+idx, 25, f'{idx}', color_pair | curses.A_BOLD | curses.A_UNDERLINE)
        stdscr.addstr(1+idx, 26, f': {probablity:5.2f}%', color_pair)
        stdscr.addstr(1+idx, 35, f'|{bar}|', color_pair)

    stdscr.refresh()


def main():
    stdscr, = curses_init()

    try:
        stdscr.erase()

        y = np.array([3.27556542e-12, 1.28491850e-17, 3.04990157e-14, 6.83018968e-10, 9.31368025e-09,
                      4.84108503e-11, 8.83054457e-20, 5.98653244e-07, 6.97749520e-05, 9.99929616e-01])

        curses_prediction_result(stdscr, *(9, y))

        time.sleep(1)
        stdscr.getch()

    except KeyboardInterrupt:
        pass
    finally:
        curses_cleanup(stdscr)


__name__ == '__main__' and main()



