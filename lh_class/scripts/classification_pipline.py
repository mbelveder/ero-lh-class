#!/usr/bin/env python3

import subprocess


def main():

    print()
    print('Welcome to the LH classification pipline!', '\n')
    print(': ' * 10)
    print('STEP 1/4')
    print(': ' * 10, '\n')

    subprocess.run('lh-class', shell=True, check=True)

    print()
    print(': ' * 10)
    print('STEP 2/4')
    print(': ' * 10, '\n')

    subprocess.run('lh-srgz-prep', shell=True, check=True)

    print()
    print(': ' * 10)
    print('STEP 3/4')
    print(': ' * 10, '\n')

    subprocess.run('lh-srgz-spec', shell=True, check=True)

    print()
    print(': ' * 10)
    print('STEP 4/4')
    print(': ' * 10, '\n')

    subprocess.run('lh-postprocess', shell=True, check=True)

    print()
    print(': ' * 10)
    print('DONE')
    print(': ' * 10, '\n')

if __name__ == '__main__':
    main()
