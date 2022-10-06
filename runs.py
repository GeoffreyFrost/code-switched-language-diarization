import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cmds-path', default='final_runs.txt', type=str)
    args = parser.parse_args()

    f = open(args.cmds_path, 'r')
    cmds = f.readlines()
    for cmd in cmds:
        print(cmd)
        os.system(cmd)

