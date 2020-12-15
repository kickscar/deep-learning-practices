import curses
import time


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
        win.addstr(0, pos, f'{display}')
        win.refresh()


curses.initscr()
percentage()
# curses.endwin()
