#!/usr/bin/env python

"""
Parse training log

Competitor to parse_log.sh
"""

import os
import re
import extract_seconds
import argparse
import csv


def get_line_type(line):
    """Return either 'test' or 'train' depending on line type
    """
    line_type = None
    if line.find('Train') != -1:
        line_type = 'train'
    elif line.find('Test') != -1:
        line_type = 'test'
    return line_type

def parse_log(path_to_log):
    """Parse log file
    Returns (train_dict_list, train_dict_names, test_dict_list, test_dict_names)

    train_dict_list and test_dict_list are lists of dicts that define the table
    rows

    train_dict_names and test_dict_names are ordered tuples of the column names
    for the two dict_lists
    """

    re_iteration = re.compile('Iteration (\d+)')
    re_accuracy = re.compile('output #\d+: accuracy = ([\.\d]+)')
    re_loss = re.compile('Iteration \d+, loss = ([\.\d]+)')
    re_lr = re.compile('lr = ([\d]+e-[\d]+|[\.\d]+)')

    # Pick out lines of interest
    iteration = -1
    current_line_iteration = -1
    iteration_type = 'train'

    accuracy = -1
    learning_rate = float('NaN')

    train_dict_list = []
    test_dict_list = []
    train_dict_names = ('Iterations', 'Seconds', 'Loss', 'LearningRate')
    test_dict_names = ('Iterations', 'Seconds', 'Loss', 'Accuracy')

    logfile_year = extract_seconds.get_log_created_year(path_to_log)
    with open(path_to_log) as f:
        start_time = extract_seconds.get_start_time(f, logfile_year)

        for line in f:
            iteration_match = re_iteration.search(line)
            if iteration_match:
                current_line_iteration = int(iteration_match.group(1))
            if current_line_iteration == -1:
                # Only look for other stuff if we've found the first iteration
                continue

            if(iteration < current_line_iteration):

                iteration = current_line_iteration

                # new iteration
                if(iteration > 0):
                    # log previous iteration
                    if(iteration_type == 'train'):
                        train_dict_list.append(iteration_dict)
                    else:
                        test_dict_list.append(iteration_dict)


                time = extract_seconds.extract_datetime_from_line(line,
                                                              logfile_year)
                seconds = (time - start_time).total_seconds()
                iteration_dict = {'Iterations': '{:d}'.format(iteration),
                                 'Seconds': '{:f}'.format(seconds)}
                iteration_type = 'train'



            if get_line_type(line) == 'test':
                iteration_type = 'test'

            lr_match = re_lr.search(line)
            if lr_match:
                iteration_dict['LearningRate'] = float(lr_match.group(1))

            accuracy_match = re_accuracy.search(line)
            if accuracy_match:
                iteration_dict['Accuracy'] = float(accuracy_match.group(1))

            loss_match = re_loss.search(line)
            if loss_match:
                 iteration_dict['Loss'] = float(loss_match.group(1))


        # log last iteration
        if(iteration_type == 'train'):
            train_dict_list.append(iteration_dict)
        else:
            test_dict_list.append(iteration_dict)

    return train_dict_list, train_dict_names, test_dict_list, test_dict_names


def save_csv_files(logfile_path, output_dir, train_dict_list, train_dict_names,
                   test_dict_list, test_dict_names, verbose=False):
    """Save CSV files to output_dir

    If the input log file is, e.g., caffe.INFO, the names will be
    caffe.INFO.train and caffe.INFO.test
    """

    log_basename = os.path.basename(logfile_path)
    train_filename = os.path.join(output_dir, log_basename + '.train')
    write_csv(train_filename, train_dict_list, train_dict_names, verbose)

    test_filename = os.path.join(output_dir, log_basename + '.test')
    write_csv(test_filename, test_dict_list, test_dict_names, verbose)


def write_csv(output_filename, dict_list, header_names, verbose=False):
    """Write a CSV file
    """

    with open(output_filename, 'w') as f:
        dict_writer = csv.DictWriter(f, header_names, extrasaction='ignore')
        dict_writer.writeheader()
        dict_writer.writerows(dict_list)
    if verbose:
        print('Wrote {0}'.format(output_filename))


def parse_args():
    description = ('Parse a Caffe training log into two CSV files '
                   'containing training and testing information')
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('logfile_path',
                        help='Path to log file')

    parser.add_argument('output_dir',
                        help='Directory in which to place output CSV files')

    parser.add_argument('--verbose',
                        action='store_true',
                        help='Print some extra info (e.g., output filenames)')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    train_dict_list, train_dict_names, test_dict_list, test_dict_names = \
        parse_log(args.logfile_path)
    save_csv_files(args.logfile_path, args.output_dir, train_dict_list,
                   train_dict_names, test_dict_list, test_dict_names)


if __name__ == '__main__':
    main()
