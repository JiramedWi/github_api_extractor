import os
print("helloworld!")
with open(os.path.abspath('example.txt'), 'w') as file:
    # Write content to the file
    file.write('Hello, World!\n')
    file.write('This is a second line.')

print('File written successfully.')