import time


# Print iterations progress
def print_progress_bar(cur_progress, max_progress=100, length=100, prefix='', suffix='', fill='█'):

    # percent = ("{0:." + str(decimals) + "f}").format(100 * (progress / float(max)))
    percent = int(100 * (cur_progress / float(max_progress)))

    filled_length = int(length * cur_progress // max_progress)
    bar = fill * filled_length + '-' * (length - filled_length)

    print(f'{prefix} |{bar}| {percent:3d}% {suffix}', end='\r')

    cur_progress == max_progress and print('')


# items = list(range(0, 57))
count = 600

# Initial call to print 0% progress
# print_progress_bar(0, length, prefix='Progress:', suffix='Complete', length=50)

for i in range(count+1):

    # Do stuff...
    time.sleep(0.05)

    # Update Progress Bar
    print_progress_bar(i, max_progress=count, length=30, prefix='Progress:', suffix='Complete')


for i in range(count+1):

    # Do stuff...
    time.sleep(0.05)

    # Update Progress Bar
    print_progress_bar(i, max_progress=count, length=30, prefix='Progress:', suffix='Complete')
