#!/usr/bin/env python3

import subprocess
import argparse

INPUT_HELPSTR = 'Please enter the path to the input data. \
It can be downloaded here: https://disk.yandex.ru/d/UybgtHAwTIdaWA'
OUTPUT_HELPSTR = 'Please enter the path to the output directory.'

parser = argparse.ArgumentParser()  # create a parser

# specify the arguments
# TODO: make shure that the link is correct before the release
parser.add_argument(
    '-input_path', '-i', type=str, required=True, help=INPUT_HELPSTR
    )
parser.add_argument(
    '-output_path', '-o', type=str, required=True, help=OUTPUT_HELPSTR
    )

args = parser.parse_args()  # parse all the arguments
input_path = args.input_path
output_path = args.output_path


def main():

    print()
    print('Welcome to the eROSITA Lockman Hole classification pipline!', '\n')
    print(': ' * 10)
    print('STEP 1/4')
    print(': ' * 10, '\n')

    subprocess.run(
        f'lh-class -i {input_path} -o {output_path}',
        shell=True, check=True
        )

    print()
    print(': ' * 10)
    print('STEP 2/4')
    print(': ' * 10, '\n')

    subprocess.run(
        f'lh-srgz-prep -i {input_path} -o {output_path}',
        shell=True, check=True
        )

    print()
    print(': ' * 10)
    print('STEP 3/4')
    print(': ' * 10, '\n')

    subprocess.run(
        f'lh-srgz-spec -i {input_path} -o {output_path}',
        shell=True, check=True
        )

    print()
    print(': ' * 10)
    print('STEP 4/4')
    print(': ' * 10, '\n')

    subprocess.run(
        f'lh-postprocess -i {input_path} -o {output_path}',
        shell=True, check=True
        )

    print()
    print(': ' * 10)
    print('DONE')
    print(': ' * 10, '\n')


if __name__ == '__main__':
    main()
