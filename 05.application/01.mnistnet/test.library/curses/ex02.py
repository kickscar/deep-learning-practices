import time
import curses


def percentage():
    win = curses.newwin(0, 0, 0, 0)
    # win.border(0)
    loading = 0
    while loading < 100:
        loading += 1
        time.sleep(0.03)
        update_progress(win, loading)


def update_progress(win, progress):
    rangex = (30 / float(100)) * progress
    pos = int(rangex)
    display = '#'
    if pos != 0:
        win.addstr(0, pos, f'001/600{display}')
        win.refresh()


def main(stdscr):
    # stdscr.clear()

    # percentage()
    win = curses.newwin(0, 0, 0, 0)

    for i in range(10):
        win.addstr(0, 1, f'epoch {i:02d}/20')
        win.refresh()
        time.sleep(0.5)

    # stdscr.refresh()
    # stdscr.getkey()


# curses.wrapper(main)
stdscr = curses.initscr()
curses.noecho()
curses.cbreak()
stdscr.keypad(True)
# stdscr.scrollok(True)

curses.curs_set(False)

win = curses.newwin(0, 0, 0, 0)
# win.border(0)

# mypad = curses.newpad(40, 60)

for i in range(400):

    win.addstr(i*2, 0, f'epoch {i+1:02d}/20')

    for j in range(1, 10):
        win.addstr(i*2+1, 0, f'{j:03d}/50')
        win.refresh()
        # mypad.refresh(i, 0, 5, 5, 10, 60)
        time.sleep(0.1)
        # stdscr.scroll()

curses.nocbreak()
stdscr.keypad(False)
curses.echo()

# curses.endwin()
