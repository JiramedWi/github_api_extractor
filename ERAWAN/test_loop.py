import time
import requests

url = 'https://notify-api.line.me/api/notify'
token = 'nHKxy92Z03QXUNvN3jfc61AV6fnPgrPC1cVuxeqWzE0'
headers = {'content-type': 'application/x-www-form-urlencoded', 'Authorization': 'Bearer ' + token}

msg = 'Hello LINE Notify'

# Number of loops you want to run
num_loops = 5

# Number of max loops you want to run
max_loops = 10

# Counter for the number of loops completed
loop_count = 0

# Time to pause in seconds (1 hour = 3600 seconds)
pause_time = 3

while True:
    # Your code for each loop iteration goes here
    print(f'Loop {loop_count + 1}')

    # Increment the loop count
    loop_count += 1
    r = requests.post(url, headers=headers, data={'message': msg})
    print(r.text)

    # Pause for 1 hour after every num_loops iterations
    if loop_count % num_loops == 0:
        wait = 'Pausing for 3 secs...'
        r = requests.post(url, headers=headers, data={'message': wait})
        print(r.text)
        time.sleep(pause_time)
        resume = 'Resuming...'
        r = requests.post(url, headers=headers, data={'message': resume})
        print(r.text)

    if loop_count == max_loops:
        done = f'all loop done at {max_loops}'
        r = requests.post(url, headers=headers, data={'message': done})
        print(r.text)
        break
