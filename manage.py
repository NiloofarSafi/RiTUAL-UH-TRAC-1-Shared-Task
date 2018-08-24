# -*- coding: utf-8 -*-

from __future__ import print_function
import click

from config import config
import os
import sys
import logging


logging.basicConfig( stream=sys.stderr, level=logging.DEBUG )
# Set the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

app = config[os.getenv('CYBERBULLYING_CONFIG') or 'default']



@click.group()
def manager():
    pass


@manager.command()
@click.option('--count', default=1, help='number of greetings')
@click.argument('name')
def classification(count, name):
    print("Inside the command")
    print(count)
    print(name)



@manager.command()
@click.option('--train', help='train set file')
@click.option('--test', help='test set file')
@click.option('--classifier', help='[SVM|LR]')

def runexpser(train, test, classifier):
    from tasks import run_experiment
    for feature in app.FEATURES:
        print ("Feature : %s"%feature)
        run_experiment('Experiment_', train, test, feature, classifier)
        print ("done")



@manager.command()
@click.option('--input', help='/path/to/corpus/file')
def display_result(input):
    from utils import display_classification_results
    display_classification_results(input)

if __name__ == "__main__":
    manager()
