#!/usr/bin/env python3

import subprocess
import argparse  # import argparse

INPUT_HELPSTR = 'Please enter the path to the input data. \
It can be downloaded here: https://disk.yandex.ru/d/UybgtHAwTIdaWA'
OUTPUT_HELPSTR = 'Please enter the path to the output data.'

parser = argparse.ArgumentParser()  # create a parser

# specify the arguments
# TODO: make shure that the link is correct before the release
parser.add_argument(
    '-input_path', '-i', type=str, required=True, help=INPUT_HELPSTR
    )
parser.add_argument(
    '-output_path', '-o', type=str, required=True, help=OUTPUT_HELPSTR
    )


def main():

    args = parser.parse_args()  # parse all the arguments

    print(args.input_path)
    print(args.output_path)

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
